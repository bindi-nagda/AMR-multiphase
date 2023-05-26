// ---------------------------------------------------------------------
//
// Copyright (c) 2015 - 2020 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibtk/CCPoissonSolverManager.h"
#include "ibtk/CartCellDoubleCubicCoarsen.h"
#include "ibtk/CartCellDoubleQuadraticCFInterpolation.h"
#include "ibtk/CartCellRobinPhysBdryOp.h"
#include "ibtk/CellNoCornersFillPattern.h"
#include "ibtk/CoarseFineBoundaryRefinePatchStrategy.h"
#include "ibtk/FACPreconditionerStrategy.h"
#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/HierarchyMathOps.h"
#include "ibtk/LinearSolver.h"
#include "ibtk/PETScKrylovLinearSolver.h"
#include "ibtk/PETScLevelSolver.h"
#include "ibtk/PoissonSolver.h"
#include "ibtk/RobinPhysBdryPatchStrategy.h"
#include "ibtk/SideNoCornersFillPattern.h"
#include "ibtk/ibtk_utilities.h"
#include "ibtk/namespaces.h" // IWYU pragma: keep
#include <ibtk/CartCellDoubleQuadraticCFInterpolation.h>
#include <ibtk/CartSideDoubleQuadraticCFInterpolation.h>
#include <ibtk/SideSynchCopyFillPattern.h>

#include "Box.h"
#include "BoxList.h"
#include "CellData.h"
#include "CellDataFactory.h"
#include "CellVariable.h"
#include "MultiblockDataTranslator.h"
#include "Patch.h"
#include "PatchDescriptor.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "PoissonSpecifications.h"
#include "ProcessorMapping.h"
#include "SAMRAIVectorReal.h"
#include "VariableFillPattern.h"
#include "tbox/Array.h"
#include "tbox/Database.h"
#include "tbox/MemoryDatabase.h"
#include "tbox/Pointer.h"
#include "tbox/Timer.h"
#include "tbox/Utilities.h"

#include "petscksp.h"

#include <Eigen/LU>

#include <SideGeometry.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

// Local includes
#include "VCTwoFluidStaggeredStokesBoxRelaxationFACOperator.h"

// FORTRAN ROUTINES
#define R_B_G_S IBTK_FC_FUNC_(rbgs, RBGS)

extern "C"
{
    void R_B_G_S(const double*, // dx
                 const int&,    // ilower0
                 const int&,    // iupper0
                 const int&,    // ilower1
                 const int&,    // iupper1
                 double* const, // un_data_0
                 double* const, // un_data_1
                 const int&,    // un_gcw
                 double* const, // us_data_0
                 double* const, // us_data_0
                 const int&,    // us_gcw
                 double* const, // p_data_
                 const int&,    // p_gcw
                 double* const, // f_p_data
                 const int&,    // f_p_gcw
                 double* const, // f_un_data_0
                 double* const, // f_un_data_1
                 const int&,    // f_un_gcw
                 double* const, // f_us_data_0
                 double* const, // f_us_data_1
                 const int&,    // f_us_gcw
                 double* const, // thn_data
                 const int&,    // thn_gcw
                 const double&, // eta_n    // whatever will be passed in will be treated as a reference to a double
                 const double&, // eta_s    // telling the compiler that the function is expecting a reference
                 const double&, // nu_n
                 const double&, // nu_s
                 const double&, // xi
                 const double&, // w = under relaxation factor
                 const double&, // C in C*u term
                 const double&, // D
                 const int&);   // red_or_black
}
/////////////////////////////// NAMESPACE ////////////////////////////////////
namespace IBTK
{
/////////////////////////////// STATIC ///////////////////////////////////////
namespace
{
// Timers.
static Timer* t_smooth_error;
static Timer* t_solve_coarsest_level;
static Timer* t_compute_residual;

// Default data depth.
static const int DEFAULT_DATA_DEPTH = 1;

// Number of ghosts cells used for each variable quantity.
static const int CELLG = 1;
static const int SIDEG = 1;

// Types of refining and coarsening to perform prior to setting coarse-fine
// boundary and physical boundary ghost cell values.
static const std::string CC_DATA_REFINE_TYPE =
    "CONSERVATIVE_LINEAR_REFINE"; // how to fill in fine cells from coarse cells, how to fill ghost cells on refine
                                  // patch
static const std::string SC_DATA_REFINE_TYPE =
    "CONSERVATIVE_LINEAR_REFINE"; // how to fill in fine cells from coarse cells, how to fill ghost cells on refine
                                  // patch
static const std::string DATA_REFINE_TYPE = "NONE";
static const bool USE_CF_INTERPOLATION = true;
static const std::string DATA_COARSEN_TYPE = "NONE";

// Type of extrapolation to use at physical boundaries; used only to evaluate
// composite grid residuals.
static const std::string BDRY_EXTRAP_TYPE = "LINEAR";

// Whether to enforce consistent interpolated values at Type 2 coarse-fine
// interface ghost cells; used only to evaluate composite grid residuals.
static const bool CONSISTENT_TYPE_2_BDRY = false;
} // namespace
double convertToThs(double Thn);
/////////////////////////////// PUBLIC ///////////////////////////////////////

VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::VCTwoFluidStaggeredStokesBoxRelaxationFACOperator(
    const std::string& object_name,
    const std::string& default_options_prefix)
    : FACPreconditionerStrategy(object_name)
{
    // Create variables and register them with the variable database.
    VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
    Pointer<VariableContext> d_ctx = var_db->getContext(d_object_name + "::context");
    Pointer<VariableContext> scr_ctx = var_db->getContext(d_object_name + "::scr_context");

    // State scratch variables: Velocity and pressure.
    // Prepend with d_object_name to ensure there are no conflicts.
    Pointer<SideVariable<NDIM, double>> un_scr_var = new SideVariable<NDIM, double>(d_object_name + "::un_scr");
    Pointer<SideVariable<NDIM, double>> us_scr_var = new SideVariable<NDIM, double>(d_object_name + "::us_scr");
    Pointer<CellVariable<NDIM, double>> p_scr_var = new CellVariable<NDIM, double>(d_object_name + "::p_scr");

    // Register patch data indices
    if (var_db->checkVariableExists(d_object_name + "::un_scr"))
    {
        un_scr_var = var_db->getVariable(d_object_name + "::un_scr");
        d_un_scr_idx = var_db->mapVariableAndContextToIndex(un_scr_var, d_ctx);
        var_db->removePatchDataIndex(d_un_scr_idx);
        d_scratch_idxs[0] = var_db->mapVariableAndContextToIndex(un_scr_var, scr_ctx);
        var_db->removePatchDataIndex(d_scratch_idxs[0]);
    }
    if (var_db->checkVariableExists(d_object_name + "::us_scr"))
    {
        us_scr_var = var_db->getVariable(d_object_name + "::us_scr");
        d_us_scr_idx = var_db->mapVariableAndContextToIndex(us_scr_var, d_ctx);
        var_db->removePatchDataIndex(d_us_scr_idx);
        d_scratch_idxs[1] = var_db->mapVariableAndContextToIndex(us_scr_var, scr_ctx);
        var_db->removePatchDataIndex(d_scratch_idxs[1]);
    }
    if (var_db->checkVariableExists(d_object_name + "::p_scr"))
    {
        p_scr_var = var_db->getVariable(d_object_name + "::p_scr");
        d_p_scr_idx = var_db->mapVariableAndContextToIndex(p_scr_var, d_ctx);
        var_db->removePatchDataIndex(d_p_scr_idx);
        d_scratch_idxs[2] = var_db->mapVariableAndContextToIndex(p_scr_var, scr_ctx);
        var_db->removePatchDataIndex(d_scratch_idxs[2]);
    }
    d_un_scr_idx = var_db->registerVariableAndContext(un_scr_var, d_ctx, IntVector<NDIM>(1));
    d_us_scr_idx = var_db->registerVariableAndContext(us_scr_var, d_ctx, IntVector<NDIM>(1));
    d_p_scr_idx = var_db->registerVariableAndContext(p_scr_var, d_ctx, IntVector<NDIM>(1));
    d_scratch_idxs[0] = var_db->registerVariableAndContext(un_scr_var, scr_ctx, IntVector<NDIM>(1));
    d_scratch_idxs[1] = var_db->registerVariableAndContext(us_scr_var, scr_ctx, IntVector<NDIM>(1));
    d_scratch_idxs[2] = var_db->registerVariableAndContext(p_scr_var, scr_ctx, IntVector<NDIM>(1));

    // Setup Timers.
    IBTK_DO_ONCE(t_smooth_error = TimerManager::getManager()->getTimer(
                     "IBTK::VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::smoothError()");
                 t_solve_coarsest_level = TimerManager::getManager()->getTimer(
                     "IBTK::VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::solveCoarsestLevel()");
                 t_compute_residual = TimerManager::getManager()->getTimer(
                     "IBTK::VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::computeResidual()"););
    return;
}

VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::~VCTwoFluidStaggeredStokesBoxRelaxationFACOperator()
{
    // Dallocate operator state first
    deallocateOperatorState();
    return;
}

// create another member function to set-up Thn
// Thn is defined in the input file and read in using muParserCartGridFunction
void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::setThnIdx(int thn_idx)
{
    d_thn_idx = thn_idx;
    return;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::restrictResidual(const SAMRAIVectorReal<NDIM, double>& src,
                                                                    SAMRAIVectorReal<NDIM, double>& dst,
                                                                    const int dst_ln)
{
    // Pull out patch indices.
    const int dst_un_idx = dst.getComponentDescriptorIndex(0);
    const int dst_us_idx = dst.getComponentDescriptorIndex(1);
    const int dst_p_idx = dst.getComponentDescriptorIndex(2);
    std::array<int, 3> dst_idxs = { dst_un_idx, dst_us_idx, dst_p_idx };

    const int src_un_idx = src.getComponentDescriptorIndex(0);
    const int src_us_idx = src.getComponentDescriptorIndex(1);
    const int src_p_idx = src.getComponentDescriptorIndex(2);
    std::array<int, 3> src_idxs = { src_un_idx, src_us_idx, src_p_idx };

    // SAMRAI's refine operators will copy data from patch interiors if the patch indices are different.
    // Therefore, I don't think we need to do this
    // TODO: test if this is necessary.
    if (dst_un_idx != src_un_idx)
    {
        HierarchySideDataOpsReal<NDIM, double> level_sc_data_ops(d_hierarchy, dst_ln, dst_ln);
        level_sc_data_ops.copyData(dst_un_idx, src_un_idx, false /*interior_only*/);
    }
    if (dst_us_idx != src_us_idx)
    {
        HierarchySideDataOpsReal<NDIM, double> level_sc_data_ops(d_hierarchy, dst_ln, dst_ln);
        level_sc_data_ops.copyData(dst_us_idx, src_us_idx, false /*interior_only*/);
    }
    if (dst_p_idx != src_p_idx)
    {
        HierarchyCellDataOpsReal<NDIM, double> level_cc_data_ops(d_hierarchy, dst_ln, dst_ln);
        level_cc_data_ops.copyData(dst_p_idx, src_p_idx, false /*interior_only*/);
    }

    // Now perform restriction
    performRestriction(dst_idxs, src_idxs, dst_ln);
    return;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::prolongError(const SAMRAIVectorReal<NDIM, double>& src,
                                                                SAMRAIVectorReal<NDIM, double>& dst,
                                                                const int dst_ln)
{
    // Pull out patch indices.
    const int dst_un_idx = dst.getComponentDescriptorIndex(0);
    const int dst_us_idx = dst.getComponentDescriptorIndex(1);
    const int dst_p_idx = dst.getComponentDescriptorIndex(2);
    std::array<int, 3> dst_idxs = { dst_un_idx, dst_us_idx, dst_p_idx };

    const int src_un_idx = src.getComponentDescriptorIndex(0);
    const int src_us_idx = src.getComponentDescriptorIndex(1);
    const int src_p_idx = src.getComponentDescriptorIndex(2);
    std::array<int, 3> src_idxs = { src_un_idx, src_us_idx, src_p_idx };

    // Now perform prolongation
    performProlongation(dst_idxs, src_idxs, dst_ln);
    return;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::prolongErrorAndCorrect(const SAMRAIVectorReal<NDIM, double>& src,
                                                                          SAMRAIVectorReal<NDIM, double>& dst,
                                                                          const int dst_ln)
{
    // Pull out patch indices.
    const int dst_un_idx = dst.getComponentDescriptorIndex(0);
    const int dst_us_idx = dst.getComponentDescriptorIndex(1);
    const int dst_p_idx = dst.getComponentDescriptorIndex(2);
    std::array<int, 3> dst_idxs = { d_un_scr_idx, d_us_scr_idx, d_p_scr_idx };

    const int src_un_idx = src.getComponentDescriptorIndex(0);
    const int src_us_idx = src.getComponentDescriptorIndex(1);
    const int src_p_idx = src.getComponentDescriptorIndex(2);
    std::array<int, 3> src_idxs = { src_un_idx, src_us_idx, src_p_idx };

    // I'm not sure why/if we need to do this, but this was done in other implementations...
    // TODO: Test if this is necessary.
    if (dst_un_idx != src_un_idx)
    {
        HierarchySideDataOpsReal<NDIM, double> level_sc_data_ops(d_hierarchy, dst_ln - 1, dst_ln - 1);
        level_sc_data_ops.add(dst_un_idx, dst_un_idx, src_un_idx, false /*interior_only*/);
    }
    if (dst_us_idx != src_us_idx)
    {
        HierarchySideDataOpsReal<NDIM, double> level_sc_data_ops(d_hierarchy, dst_ln - 1, dst_ln - 1);
        level_sc_data_ops.add(dst_us_idx, dst_us_idx, src_us_idx, false /*interior_only*/);
    }
    if (dst_p_idx != src_p_idx)
    {
        HierarchyCellDataOpsReal<NDIM, double> level_cc_data_ops(d_hierarchy, dst_ln - 1, dst_ln - 1);
        level_cc_data_ops.add(dst_p_idx, dst_p_idx, src_p_idx, false /*interior_only*/);
    }

    // Now prolong and correct data
    performProlongation(dst_idxs, src_idxs, dst_ln);
    HierarchySideDataOpsReal<NDIM, double> level_sc_data_ops(d_hierarchy, dst_ln, dst_ln);
    level_sc_data_ops.add(dst_un_idx, dst_un_idx, d_un_scr_idx, false /*interior_only*/);
    level_sc_data_ops.add(dst_us_idx, dst_us_idx, d_us_scr_idx, false /*interior_only*/);
    HierarchyCellDataOpsReal<NDIM, double> level_cc_data_ops(d_hierarchy, dst_ln, dst_ln);
    level_cc_data_ops.add(dst_p_idx, dst_p_idx, d_p_scr_idx, false /*interior_only*/);
    return;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::smoothError(
    SAMRAIVectorReal<NDIM, double>& error,          // Solution
    const SAMRAIVectorReal<NDIM, double>& residual, // RHS - A*error on the entire patch level
    int level_num,
    int num_sweeps,
    bool /*performing_pre_sweeps*/,
    bool /*performing_post_sweeps*/)
{
    if (num_sweeps == 0) return;

    IBTK_TIMER_START(t_smooth_error);

    // Get the vector components. These pull out patch data indices
    const int un_idx = error.getComponentDescriptorIndex(0);      // network velocity, Un
    const int us_idx = error.getComponentDescriptorIndex(1);      // solvent velocity, Us
    const int P_idx = error.getComponentDescriptorIndex(2);       // pressure
    const int thn_idx = d_thn_idx;
    const int f_un_idx = residual.getComponentDescriptorIndex(0); // RHS_Un
    const int f_us_idx = residual.getComponentDescriptorIndex(1); // RHS_Us
    const int f_P_idx = residual.getComponentDescriptorIndex(2);  // RHS_pressure

    d_hierarchy = error.getPatchHierarchy();
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(level_num);

    const int un_scratch_idx = d_scratch_idxs[0];
    const int us_scratch_idx = d_scratch_idxs[1];
    const int p_scratch_idx = d_scratch_idxs[2];
    // Cache coarse-fine interface ghost cell values in the "scratch" data.
    // TODO: Determine if we need to do this.
    if (level_num > 0 && num_sweeps > 1)
    {
        int patch_counter = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_counter)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<SideData<NDIM, double>> error_un_data = error.getComponentPatchData(0, *patch);
            Pointer<SideData<NDIM, double>> scratch_un_data = patch->getPatchData(un_scratch_idx);
            Pointer<SideData<NDIM, double>> error_us_data = error.getComponentPatchData(1, *patch);
            Pointer<SideData<NDIM, double>> scratch_us_data = patch->getPatchData(us_scratch_idx);
            for (unsigned int axis = 0; axis < NDIM; ++axis)
            {
                scratch_un_data->getArrayData(axis).copy(error_un_data->getArrayData(axis),
                                                         d_patch_side_bc_box_overlap[level_num][patch_counter][axis],
                                                         IntVector<NDIM>(0));
                scratch_us_data->getArrayData(axis).copy(error_us_data->getArrayData(axis),
                                                         d_patch_side_bc_box_overlap[level_num][patch_counter][axis],
                                                         IntVector<NDIM>(0));
            }

            Pointer<CellData<NDIM, double>> error_p_data = error.getComponentPatchData(2, *patch);
            Pointer<CellData<NDIM, double>> scratch_p_data = patch->getPatchData(p_scratch_idx);
            scratch_p_data->getArrayData().copy(error_p_data->getArrayData(),
                                                d_patch_cell_bc_box_overlap[level_num][patch_counter],
                                                IntVector<NDIM>(0));
        }
    }

    // outer for loop for number of sweeps
    for (int sweep = 0; sweep < 2 * num_sweeps; sweep++)
    {
        // Fill in ghost cells. We only want to use values on our current level to fill in ghost cells.
        // TODO: d_ghostfill_no_restrict_scheds does not fill in ghost cells in at coarse fine interfaces. We need to
        // set that up. One way of doing that is using the existing operators in IBAMR to compute the "normal
        // extension."
        if (level_num > 0)
        {
            if (sweep > 0)
            {
                // Copy the coarse-fine interface ghost cell values which are
                // cached in the scratch data into the error data.
                // TODO: Determine if we actually need to do this.
                int patch_counter = 0;
                for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_counter)
                {
                    Pointer<Patch<NDIM>> patch = level->getPatch(p());
                    Pointer<SideData<NDIM, double>> error_un_data = error.getComponentPatchData(0, *patch);
                    Pointer<SideData<NDIM, double>> scratch_un_data = patch->getPatchData(un_scratch_idx);
                    Pointer<SideData<NDIM, double>> error_us_data = error.getComponentPatchData(1, *patch);
                    Pointer<SideData<NDIM, double>> scratch_us_data = patch->getPatchData(us_scratch_idx);
                    for (unsigned int axis = 0; axis < NDIM; ++axis)
                    {
                        error_un_data->getArrayData(axis).copy(
                            scratch_un_data->getArrayData(axis),
                            d_patch_side_bc_box_overlap[level_num][patch_counter][axis],
                            IntVector<NDIM>(0));
                        error_us_data->getArrayData(axis).copy(
                            scratch_us_data->getArrayData(axis),
                            d_patch_side_bc_box_overlap[level_num][patch_counter][axis],
                            IntVector<NDIM>(0));
                    }

                    Pointer<CellData<NDIM, double>> error_p_data = error.getComponentPatchData(2, *patch);
                    Pointer<CellData<NDIM, double>> scratch_p_data = patch->getPatchData(p_scratch_idx);
                    error_p_data->getArrayData().copy(scratch_p_data->getArrayData(),
                                                      d_patch_cell_bc_box_overlap[level_num][patch_counter],
                                                      IntVector<NDIM>(0));
                }

                performGhostFilling({ un_idx, us_idx, P_idx }, level_num);
            }
            // Compute the normal extension of the solution at coarse fine interfaces if we are not on the coarsest
            // level
            // TODO: 0 here is coarsest level number, we should set this to a variable.
            d_cc_bdry_op->setPatchDataIndex(P_idx);
            d_sc_bdry_op->setPatchDataIndices({ un_idx, us_idx });
            const IntVector<NDIM>& ratio = level->getRatioToCoarserLevel();
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                // TODO: 1 here is the ghost width to fill, this should be variable.
                const IntVector<NDIM>& gcw_to_fill = 1;
                d_sc_bdry_op->computeNormalExtension(*patch, ratio, gcw_to_fill);
                d_cc_bdry_op->computeNormalExtension(*patch, ratio, gcw_to_fill);
            }
        }
        else
        {
            performGhostFilling({ un_idx, us_idx, P_idx }, level_num);
        }

        IntVector<NDIM> xp(1, 0), yp(0, 1);

        // loop through all patches on this level
        int patch_counter = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_counter)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx(); // dx[0] -> x, dx[1] -> y
            Pointer<CellData<NDIM, double>> thn_data = patch->getPatchData(thn_idx);
            Pointer<SideData<NDIM, double>> un_data = patch->getPatchData(un_idx);
            Pointer<SideData<NDIM, double>> us_data = patch->getPatchData(us_idx);
            Pointer<CellData<NDIM, double>> p_data = patch->getPatchData(P_idx);
            Pointer<SideData<NDIM, double>> un_scr_data = patch->getPatchData(d_un_scr_idx);
            Pointer<SideData<NDIM, double>> us_scr_data = patch->getPatchData(d_us_scr_idx);
            Pointer<CellData<NDIM, double>> p_scr_data = patch->getPatchData(d_p_scr_idx);
            Pointer<SideData<NDIM, double>> f_un_data = patch->getPatchData(f_un_idx);
            Pointer<SideData<NDIM, double>> f_us_data = patch->getPatchData(f_us_idx);
            Pointer<CellData<NDIM, double>> f_p_data = patch->getPatchData(f_P_idx);

            double* const un_data_0 = un_data->getPointer(0);
            double* const un_data_1 = un_data->getPointer(1);
            double* const us_data_0 = us_data->getPointer(0);
            double* const us_data_1 = us_data->getPointer(1);
            double* const thn_ptr_data = thn_data->getPointer(0);
            double* const p_ptr_data = p_data->getPointer(0);
            double* const f_un_data_0 = f_un_data->getPointer(0);
            double* const f_un_data_1 = f_un_data->getPointer(1);
            double* const f_us_data_0 = f_us_data->getPointer(0);
            double* const f_us_data_1 = f_us_data->getPointer(1);
            double* const f_p_ptr_data = f_p_data->getPointer(0);

            const Box<NDIM>& patch_box = patch->getBox();
            const IntVector<NDIM>& patch_lower =
                patch_box.lower(); // patch_lower(0), patch_lower(1) are min indices in x and y-dir
            const IntVector<NDIM>& patch_upper =
                patch_box.upper(); // patch_upper(0), patch_upper(1) are max indices in x and y-dir

            const IntVector<NDIM>& thn_gcw = thn_data->getGhostCellWidth();
            const IntVector<NDIM>& un_gcw = un_data->getGhostCellWidth();
            const IntVector<NDIM>& us_gcw = us_data->getGhostCellWidth();
            const IntVector<NDIM>& p_gcw = p_data->getGhostCellWidth();
            const IntVector<NDIM>& f_un_gcw = f_un_data->getGhostCellWidth();
            const IntVector<NDIM>& f_us_gcw = f_us_data->getGhostCellWidth();
            const IntVector<NDIM>& f_p_gcw = f_p_data->getGhostCellWidth();

            // Copy updated values from neighboring local patches.
            // This will determine if we are doing a "patch" based smoother or a "level" based smoother.
            if (true)
            {
                for (unsigned int axis = 0; axis < NDIM; ++axis)
                {
                    for (const auto& pair : d_patch_side_neighbor_overlap[level_num][patch_counter][axis])
                    {
                        const int src_patch_num = pair.first;
                        const Box<NDIM>& overlap = pair.second;
                        Pointer<Patch<NDIM>> src_patch = level->getPatch(src_patch_num);
                        Pointer<SideData<NDIM, double>> src_un_data = error.getComponentPatchData(0, *src_patch);
                        Pointer<SideData<NDIM, double>> src_us_data = error.getComponentPatchData(1, *src_patch);
                        un_data->getArrayData(axis).copy(src_un_data->getArrayData(axis), overlap, IntVector<NDIM>(0));
                        us_data->getArrayData(axis).copy(src_us_data->getArrayData(axis), overlap, IntVector<NDIM>(0));
                    }
                }

                for (const auto& pair : d_patch_cell_neighbor_overlap[level_num][patch_counter])
                {
                    const int src_patch_num = pair.first;
                    const Box<NDIM>& overlap = pair.second;
                    Pointer<Patch<NDIM>> src_patch = level->getPatch(src_patch_num);
                    Pointer<CellData<NDIM, double>> src_p_data = error.getComponentPatchData(2, *src_patch);
                    p_data->getArrayData().copy(src_p_data->getArrayData(), overlap, IntVector<NDIM>(0));
                }
            }

            int red_or_black = sweep % 2; // red = 0 and black = 1
            R_B_G_S(dx,
                    patch_lower(0),       // ilower0
                    patch_upper(0),       // iupper0
                    patch_lower(1),       // ilower1
                    patch_upper(1),       // iupper1
                    un_data_0,
                    un_data_1,
                    un_gcw.min(),
                    us_data_0,
                    us_data_1,
                    us_gcw.min(),
                    p_ptr_data,
                    p_gcw.min(),
                    f_p_ptr_data,
                    f_p_gcw.min(),
                    f_un_data_0,
                    f_un_data_1,
                    f_un_gcw.min(),
                    f_us_data_0,
                    f_us_data_1,
                    f_us_gcw.min(),
                    thn_ptr_data,
                    thn_gcw.min(),
                    d_eta_n,
                    d_eta_s,
                    d_nu_n,
                    d_nu_s,
                    d_xi,
                    d_w,
                    d_C,
                    d_D,
                    red_or_black);

        } // patchess
    }     // num_sweeps
    // Synchronize sides
    performSynch({ un_idx, us_idx }, level_num);

    IBTK_TIMER_STOP(t_smooth_error);
    return;
}

bool
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::solveCoarsestLevel(SAMRAIVectorReal<NDIM, double>& error,
                                                                      const SAMRAIVectorReal<NDIM, double>& residual,
                                                                      int coarsest_ln)
{
    IBTK_TIMER_START(t_solve_coarsest_level);

    smoothError(error, residual, coarsest_ln, 10, false, false);

    IBTK_TIMER_STOP(t_solve_coarsest_level);
    return true;
} // solveCoarsestLevel

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::computeResidual(SAMRAIVectorReal<NDIM, double>& residual,
                                                                   const SAMRAIVectorReal<NDIM, double>& solution,
                                                                   const SAMRAIVectorReal<NDIM, double>& rhs,
                                                                   int coarsest_level_num,
                                                                   int finest_level_num)
{
    IBTK_TIMER_START(t_compute_residual);

    // Get the vector components. These pull out patch data indices
    const int un_idx = solution.getComponentDescriptorIndex(0); // network velocity, Un
    const int us_idx = solution.getComponentDescriptorIndex(1); // solvent velocity, Us
    const int P_idx = solution.getComponentDescriptorIndex(2);  // pressure
    const int rhs_un_idx = rhs.getComponentDescriptorIndex(0);  // RHS Un
    const int rhs_us_idx = rhs.getComponentDescriptorIndex(1);  // RHS Us
    const int rhs_P_idx = rhs.getComponentDescriptorIndex(2);   // RHS P
    const int res_un_idx = residual.getComponentDescriptorIndex(0);
    const int res_us_idx = residual.getComponentDescriptorIndex(1);
    const int res_P_idx = residual.getComponentDescriptorIndex(2);
    const int thn_idx = d_thn_idx;

    d_un_fill_pattern = new SideNoCornersFillPattern(SIDEG, false, false, true);
    d_us_fill_pattern = new SideNoCornersFillPattern(SIDEG, false, false, true);
    d_P_fill_pattern = new CellNoCornersFillPattern(CELLG, false, false, true);
    // Simultaneously fill ghost cell values for all components.
    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<InterpolationTransactionComponent> transaction_comps(3);
    transaction_comps[0] = InterpolationTransactionComponent(un_idx,
                                                             un_idx,
                                                             SC_DATA_REFINE_TYPE,
                                                             USE_CF_INTERPOLATION,
                                                             DATA_COARSEN_TYPE,
                                                             BDRY_EXTRAP_TYPE,
                                                             CONSISTENT_TYPE_2_BDRY,
                                                             d_un_bc_coefs, // modifiy?
                                                             d_un_fill_pattern);
    transaction_comps[1] = InterpolationTransactionComponent(us_idx,
                                                             us_idx,
                                                             SC_DATA_REFINE_TYPE,
                                                             USE_CF_INTERPOLATION,
                                                             DATA_COARSEN_TYPE,
                                                             BDRY_EXTRAP_TYPE,
                                                             CONSISTENT_TYPE_2_BDRY,
                                                             d_us_bc_coefs, // modify?
                                                             d_us_fill_pattern);
    transaction_comps[2] = InterpolationTransactionComponent(P_idx,
                                                             CC_DATA_REFINE_TYPE,
                                                             USE_CF_INTERPOLATION,
                                                             DATA_COARSEN_TYPE,
                                                             BDRY_EXTRAP_TYPE,
                                                             CONSISTENT_TYPE_2_BDRY,
                                                             d_P_bc_coef,
                                                             d_P_fill_pattern);

    d_hier_bdry_fill = new HierarchyGhostCellInterpolation();
    d_hier_bdry_fill->initializeOperatorState(transaction_comps, d_hierarchy, coarsest_level_num, finest_level_num);
    d_hier_bdry_fill->setHomogeneousBc(d_homogeneous_bc);
    d_hier_bdry_fill->fillData(d_solution_time); // Fills in all of the ghost cells

    for (int ln = coarsest_level_num; ln <= finest_level_num; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx(); // dx[0] -> x, dx[1] -> y
            const double* const xlow =
                pgeom->getXLower();                  // {xlow[0], xlow[1]} -> physical location of bottom left of box.
            const hier::Index<NDIM>& idx_low = patch->getBox().lower();
            Pointer<CellData<NDIM, double>> p_data = patch->getPatchData(P_idx);
            Pointer<CellData<NDIM, double>> rhs_P_data =
                patch->getPatchData(rhs_P_idx); // result of applying operator (eqn 3)
            Pointer<CellData<NDIM, double>> thn_data = patch->getPatchData(thn_idx);
            Pointer<SideData<NDIM, double>> un_data = patch->getPatchData(un_idx);
            Pointer<SideData<NDIM, double>> rhs_un_data =
                patch->getPatchData(rhs_un_idx); // result of applying operator (eqn 1)
            Pointer<SideData<NDIM, double>> us_data = patch->getPatchData(us_idx);
            Pointer<SideData<NDIM, double>> rhs_us_data =
                patch->getPatchData(rhs_us_idx); // result of applying operator (eqn 2)
            Pointer<SideData<NDIM, double>> res_un_data = patch->getPatchData(res_un_idx);
            Pointer<SideData<NDIM, double>> res_us_data = patch->getPatchData(res_us_idx);
            Pointer<CellData<NDIM, double>> res_P_data = patch->getPatchData(res_P_idx);
            IntVector<NDIM> xp(1, 0), yp(0, 1);

            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++) // cell-centers
            {
                const CellIndex<NDIM>& idx = ci();

                SideIndex<NDIM> lower_x_idx(idx, 0, 0); // (i-1/2,j)
                SideIndex<NDIM> upper_x_idx(idx, 0, 1); // (i+1/2,j)
                SideIndex<NDIM> lower_y_idx(idx, 1, 0); // (i,j-1/2)
                SideIndex<NDIM> upper_y_idx(idx, 1, 1); // (i,j+1/2)

                // thn at sidess
                double thn_lower_x = 0.5 * ((*thn_data)(idx) + (*thn_data)(idx - xp)); // thn(i-1/2,j)
                double thn_upper_x = 0.5 * ((*thn_data)(idx) + (*thn_data)(idx + xp)); // thn(i+1/2,j)
                double thn_lower_y = 0.5 * ((*thn_data)(idx) + (*thn_data)(idx - yp)); // thn(i,j-1/2)
                double thn_upper_y = 0.5 * ((*thn_data)(idx) + (*thn_data)(idx + yp)); // thn(i,j+1/2)

                // conservation of mass

                double div_un_dot_thn_dx =
                    ((thn_upper_x * (*un_data)(upper_x_idx)) - (thn_lower_x * (*un_data)(lower_x_idx))) / dx[0];
                double div_un_dot_thn_dy =
                    ((thn_upper_y * (*un_data)(upper_y_idx)) - (thn_lower_y * (*un_data)(lower_y_idx))) / dx[1];
                double div_un_thn = div_un_dot_thn_dx + div_un_dot_thn_dy;

                double div_us_dot_ths_dx = ((convertToThs(thn_upper_x) * (*us_data)(upper_x_idx)) -
                                            (convertToThs(thn_lower_x) * (*us_data)(lower_x_idx))) /
                                           dx[0];
                double div_us_dot_ths_dy = ((convertToThs(thn_upper_y) * (*us_data)(upper_y_idx)) -
                                            (convertToThs(thn_lower_y) * (*us_data)(lower_y_idx))) /
                                           dx[1];
                double div_us_ths = div_us_dot_ths_dx + div_us_dot_ths_dy;
                (*res_P_data)(idx) = (*rhs_P_data)(idx) - (div_un_thn + div_us_ths);
            }

            for (SideIterator<NDIM> si(patch->getBox(), 0); si; si++) // side-centers in x-dir
            {
                const SideIndex<NDIM>& idx = si();                    // axis = 0, (i-1/2,j)

                CellIndex<NDIM> idx_c_low = idx.toCell(0);            // (i-1,j)
                CellIndex<NDIM> idx_c_up = idx.toCell(1);             // (i,j)
                SideIndex<NDIM> lower_y_idx(idx_c_up, 1, 0);          // (i,j-1/2)
                SideIndex<NDIM> upper_y_idx(idx_c_up, 1, 1);          // (i,j+1/2)
                SideIndex<NDIM> l_y_idx(idx_c_low, 1, 0);             // (i-1,j-1/2)
                SideIndex<NDIM> u_y_idx(idx_c_low, 1, 1);             // (i-1,j+1/2)

                // thn at sides
                double thn_lower = 0.5 * ((*thn_data)(idx_c_low) + (*thn_data)(idx_c_up)); // thn(i-1/2,j)
                // thn at corners
                double thn_imhalf_jphalf =
                    0.25 * ((*thn_data)(idx_c_low) + (*thn_data)(idx_c_up) + (*thn_data)(idx_c_up + yp) +
                            (*thn_data)(idx_c_low + yp)); // thn(i-1/2,j+1/2)
                double thn_imhalf_jmhalf =
                    0.25 * ((*thn_data)(idx_c_up) + (*thn_data)(idx_c_low) + (*thn_data)(idx_c_up - yp) +
                            (*thn_data)(idx_c_low - yp)); // thn(i-1/2,j-1/2)

                // components of first row (x-component of network vel) of network equation
                double ddx_Thn_dx_un = d_eta_n / (dx[0] * dx[0]) *
                                       ((*thn_data)(idx_c_up) * ((*un_data)(idx + xp) - (*un_data)(idx)) -
                                        (*thn_data)(idx_c_low) * ((*un_data)(idx) - (*un_data)(idx - xp)));
                double ddy_Thn_dy_un = d_eta_n / (dx[1] * dx[1]) *
                                       (thn_imhalf_jphalf * ((*un_data)(idx + yp) - (*un_data)(idx)) -
                                        thn_imhalf_jmhalf * ((*un_data)(idx) - (*un_data)(idx - yp)));
                double ddy_Thn_dx_vn = d_eta_n / (dx[1] * dx[0]) *
                                       (thn_imhalf_jphalf * ((*un_data)(upper_y_idx) - (*un_data)(u_y_idx)) -
                                        thn_imhalf_jmhalf * ((*un_data)(lower_y_idx) - (*un_data)(l_y_idx)));
                double ddx_Thn_dy_vn = -d_eta_n / (dx[0] * dx[1]) *
                                       ((*thn_data)(idx_c_up) * ((*un_data)(upper_y_idx) - (*un_data)(lower_y_idx)) -
                                        (*thn_data)(idx_c_low) * ((*un_data)(u_y_idx) - (*un_data)(l_y_idx)));

                double drag_n =
                    -d_xi / d_nu_n * thn_lower * convertToThs(thn_lower) * ((*un_data)(idx) - (*us_data)(idx));
                double pressure_n = -thn_lower / dx[0] * ((*p_data)(idx_c_up) - (*p_data)(idx_c_low));
                (*res_un_data)(idx) =
                    (*rhs_un_data)(idx) -
                    (d_D * (ddx_Thn_dx_un + ddy_Thn_dy_un + ddy_Thn_dx_vn + ddx_Thn_dy_vn + drag_n + pressure_n) +
                     d_C * thn_lower * (*un_data)(idx));

                // solvent equation
                double ddx_Ths_dx_us =
                    d_eta_s / (dx[0] * dx[0]) *
                    (convertToThs((*thn_data)(idx_c_up)) * ((*us_data)(idx + xp) - (*us_data)(idx)) -
                     convertToThs((*thn_data)(idx_c_low)) * ((*us_data)(idx) - (*us_data)(idx - xp)));
                double ddy_Ths_dy_us = d_eta_s / (dx[1] * dx[1]) *
                                       (convertToThs(thn_imhalf_jphalf) * ((*us_data)(idx + yp) - (*us_data)(idx)) -
                                        convertToThs(thn_imhalf_jmhalf) * ((*us_data)(idx) - (*us_data)(idx - yp)));
                double ddy_Ths_dx_vs =
                    d_eta_s / (dx[1] * dx[0]) *
                    (convertToThs(thn_imhalf_jphalf) * ((*us_data)(upper_y_idx) - (*us_data)(u_y_idx)) -
                     convertToThs(thn_imhalf_jmhalf) * ((*us_data)(lower_y_idx) - (*us_data)(l_y_idx)));
                double ddx_Ths_dy_vs =
                    -d_eta_s / (dx[0] * dx[1]) *
                    (convertToThs((*thn_data)(idx_c_up)) * ((*us_data)(upper_y_idx) - (*us_data)(lower_y_idx)) -
                     convertToThs((*thn_data)(idx_c_low)) * ((*us_data)(u_y_idx) - (*us_data)(l_y_idx)));

                double drag_s =
                    -d_xi / d_nu_s * thn_lower * convertToThs(thn_lower) * ((*us_data)(idx) - (*un_data)(idx));
                double pressure_s = -convertToThs(thn_lower) / dx[0] * ((*p_data)(idx_c_up) - (*p_data)(idx_c_low));
                (*res_us_data)(idx) =
                    (*rhs_us_data)(idx) -
                    (d_D * (ddx_Ths_dx_us + ddy_Ths_dy_us + ddy_Ths_dx_vs + ddx_Ths_dy_vs + drag_s + pressure_s) +
                     d_C * convertToThs(thn_lower) * (*us_data)(idx));
            }

            for (SideIterator<NDIM> si(patch->getBox(), 1); si; si++) // side-centers in y-dir
            {
                const SideIndex<NDIM>& idx = si();                    // axis = 1, (i,j-1/2)

                CellIndex<NDIM> idx_c_low = idx.toCell(0);            // (i,j-1)
                CellIndex<NDIM> idx_c_up = idx.toCell(1);             // (i,j)
                SideIndex<NDIM> lower_x_idx(idx_c_up, 0, 0);          // (i-1/2,j)
                SideIndex<NDIM> upper_x_idx(idx_c_up, 0, 1);          // (i+1/2,j)
                SideIndex<NDIM> l_x_idx(idx_c_low, 0, 0);             // (i-1/2,j-1)
                SideIndex<NDIM> u_x_idx(idx_c_low, 0, 1);             // (i+1/2,j-1)

                // thn at sides
                double thn_lower = 0.5 * ((*thn_data)(idx_c_low) + (*thn_data)(idx_c_up)); // thn(i,j-1/2)

                // thn at corners
                double thn_imhalf_jmhalf =
                    0.25 * ((*thn_data)(idx_c_low) + (*thn_data)(idx_c_up) + (*thn_data)(idx_c_up - xp) +
                            (*thn_data)(idx_c_low - xp)); // thn(i-1/2,j-1/2)
                double thn_iphalf_jmhalf =
                    0.25 * ((*thn_data)(idx_c_up) + (*thn_data)(idx_c_low) + (*thn_data)(idx_c_up + xp) +
                            (*thn_data)(idx_c_low + xp)); // thn(i+1/2,j-1/2)

                // components of second row (y-component of network vel) of network equation
                double ddy_Thn_dy_un = d_eta_n / (dx[1] * dx[1]) *
                                       ((*thn_data)(idx_c_up) * ((*un_data)(idx + yp) - (*un_data)(idx)) -
                                        (*thn_data)(idx_c_low) * ((*un_data)(idx) - (*un_data)(idx - yp)));
                double ddx_Thn_dx_un = d_eta_n / (dx[0] * dx[0]) *
                                       (thn_iphalf_jmhalf * ((*un_data)(idx + xp) - (*un_data)(idx)) -
                                        thn_imhalf_jmhalf * ((*un_data)(idx) - (*un_data)(idx - xp)));
                double ddx_Thn_dy_vn = d_eta_n / (dx[1] * dx[0]) *
                                       (thn_iphalf_jmhalf * ((*un_data)(upper_x_idx) - (*un_data)(u_x_idx)) -
                                        thn_imhalf_jmhalf * ((*un_data)(lower_x_idx) - (*un_data)(l_x_idx)));
                double ddy_Thn_dx_vn = -d_eta_n / (dx[0] * dx[1]) *
                                       ((*thn_data)(idx_c_up) * ((*un_data)(upper_x_idx) - (*un_data)(lower_x_idx)) -
                                        (*thn_data)(idx_c_low) * ((*un_data)(u_x_idx) - (*un_data)(l_x_idx)));

                double drag_n =
                    -d_xi / d_nu_n * thn_lower * convertToThs(thn_lower) * ((*un_data)(idx) - (*us_data)(idx));
                double pressure_n = -thn_lower / dx[0] * ((*p_data)(idx_c_up) - (*p_data)(idx_c_low));
                (*res_un_data)(idx) =
                    (*rhs_un_data)(idx) -
                    (d_D * (ddy_Thn_dy_un + ddx_Thn_dx_un + ddx_Thn_dy_vn + ddy_Thn_dx_vn + drag_n + pressure_n) +
                     d_C * thn_lower * (*un_data)(idx));

                // Solvent equation
                double ddy_Ths_dy_us =
                    d_eta_s / (dx[1] * dx[1]) *
                    (convertToThs((*thn_data)(idx_c_up)) * ((*us_data)(idx + yp) - (*us_data)(idx)) -
                     convertToThs((*thn_data)(idx_c_low)) * ((*us_data)(idx) - (*us_data)(idx - yp)));
                double ddx_Ths_dx_us = d_eta_s / (dx[0] * dx[0]) *
                                       (convertToThs(thn_iphalf_jmhalf) * ((*us_data)(idx + xp) - (*us_data)(idx)) -
                                        convertToThs(thn_imhalf_jmhalf) * ((*us_data)(idx) - (*us_data)(idx - xp)));
                double ddx_Ths_dy_vs =
                    d_eta_s / (dx[1] * dx[0]) *
                    (convertToThs(thn_iphalf_jmhalf) * ((*us_data)(upper_x_idx) - (*us_data)(u_x_idx)) -
                     convertToThs(thn_imhalf_jmhalf) * ((*us_data)(lower_x_idx) - (*us_data)(l_x_idx)));
                double ddy_Ths_dx_vs =
                    -d_eta_s / (dx[0] * dx[1]) *
                    (convertToThs((*thn_data)(idx_c_up)) * ((*us_data)(upper_x_idx) - (*us_data)(lower_x_idx)) -
                     convertToThs((*thn_data)(idx_c_low)) * ((*us_data)(u_x_idx) - (*us_data)(l_x_idx)));

                double drag_s =
                    -d_xi / d_nu_s * thn_lower * convertToThs(thn_lower) * ((*us_data)(idx) - (*un_data)(idx));
                double pressure_s = -convertToThs(thn_lower) / dx[0] * ((*p_data)(idx_c_up) - (*p_data)(idx_c_low));
                (*res_us_data)(idx) =
                    (*rhs_us_data)(idx) -
                    (d_D * (ddy_Ths_dy_us + ddx_Ths_dx_us + ddx_Ths_dy_vs + ddy_Ths_dx_vs + drag_s + pressure_s) +
                     d_C * convertToThs(thn_lower) * (*us_data)(idx));
            }
        }
    }

    IBTK_TIMER_STOP(t_compute_residual);
    return;
}

double
convertToThs(double Thn)
{
    return 1.0 - Thn; // Thn+Ths = 1
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::setToZero(SAMRAIVectorReal<NDIM, double>& vec, int level_num)
{
    const int un_idx = vec.getComponentDescriptorIndex(0); // network velocity, Un
    const int us_idx = vec.getComponentDescriptorIndex(1); // solvent velocity, Us
    const int P_idx = vec.getComponentDescriptorIndex(2);  // pressure
    d_hierarchy = vec.getPatchHierarchy();
    Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(level_num);
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        Pointer<SideData<NDIM, double>> un_data = patch->getPatchData(un_idx);
        Pointer<SideData<NDIM, double>> us_data = patch->getPatchData(us_idx);
        Pointer<CellData<NDIM, double>> p_data = patch->getPatchData(P_idx);
        un_data->fillAll(0.0);
        us_data->fillAll(0.0);
        p_data->fillAll(0.0);
    }
    return;
} // setToZero

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::initializeOperatorState(const SAMRAIVectorReal<NDIM, double>& sol,
                                                                           const SAMRAIVectorReal<NDIM, double>& rhs)
{
    if (d_is_initialized) deallocateOperatorState();
    // Setup solution and rhs vectors.
    Pointer<SideVariable<NDIM, double>> un_sol_var = sol.getComponentVariable(0);
    Pointer<SideVariable<NDIM, double>> us_sol_var = sol.getComponentVariable(1);
    Pointer<CellVariable<NDIM, double>> p_sol_var = sol.getComponentVariable(2);

    Pointer<SideVariable<NDIM, double>> un_rhs_var = rhs.getComponentVariable(0);
    Pointer<SideVariable<NDIM, double>> us_rhs_var = rhs.getComponentVariable(1);
    Pointer<CellVariable<NDIM, double>> p_rhs_var = rhs.getComponentVariable(2);

    d_hierarchy = sol.getPatchHierarchy();
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();

    // Rudimentary error checking.
#if !defined(NDEBUG)
    TBOX_ASSERT(un_sol_var && us_sol_var && p_sol_var);
    TBOX_ASSERT(un_rhs_var && us_rhs_var && p_rhs_var);
    TBOX_ASSERT(d_hierarchy == rhs.getPatchHierarchy());
#endif

    // Allocate scratch data
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(d_un_scr_idx, d_new_time);
        level->allocatePatchData(d_us_scr_idx, d_new_time);
        level->allocatePatchData(d_p_scr_idx, d_new_time);
        level->allocatePatchData(d_scratch_idxs[0], d_new_time);
        level->allocatePatchData(d_scratch_idxs[1], d_new_time);
        level->allocatePatchData(d_scratch_idxs[2], d_new_time);
    }

    // Cache prolongation operators. Creating refinement schedules can be expensive for hierarchies with many levels. We
    // create the schedules here, and SAMRAI will determine whether they need to be regenerated whenever we switch patch
    // indices.
    // TODO: Set the refine type via input database or setter.
    Pointer<CartesianGridGeometry<NDIM>> grid_geom = d_hierarchy->getGridGeometry();
    d_un_prolong_op = grid_geom->lookupRefineOperator(un_sol_var, "CONSERVATIVE_LINEAR_REFINE");
    d_us_prolong_op = grid_geom->lookupRefineOperator(us_sol_var, "CONSERVATIVE_LINEAR_REFINE");
    d_p_prolong_op = grid_geom->lookupRefineOperator(p_sol_var, "CONSERVATIVE_LINEAR_REFINE");
    const int un_sol_idx = sol.getComponentDescriptorIndex(0);
    const int us_sol_idx = sol.getComponentDescriptorIndex(1);
    const int p_sol_idx = sol.getComponentDescriptorIndex(2);

    RefineAlgorithm<NDIM> refine_alg;
    refine_alg.registerRefine(d_un_scr_idx, un_sol_idx, d_un_scr_idx, d_un_prolong_op);
    refine_alg.registerRefine(d_us_scr_idx, us_sol_idx, d_us_scr_idx, d_us_prolong_op);
    refine_alg.registerRefine(d_p_scr_idx, p_sol_idx, d_p_scr_idx, d_p_prolong_op);

    d_prolong_scheds.resize(finest_ln - coarsest_ln + 1);
    // Note start from zero because you can't prolong to the coarsest level.
    for (int dst_ln = coarsest_ln + 1; dst_ln <= finest_ln; ++dst_ln)
    {
        // TODO: The last argument should be the refine patch strategies. These should be, e.g. physical boundary
        // routines and fix-ups related to coarse fine interfaces.
        d_prolong_scheds[dst_ln] = refine_alg.createSchedule(d_hierarchy->getPatchLevel(dst_ln),
                                                             Pointer<PatchLevel<NDIM>>(),
                                                             dst_ln - 1,
                                                             d_hierarchy,
                                                             nullptr /* Refine patch strategy*/);
    }

    // Cache restriction operators.
    // TODO: Set the coarsen type via input database or setter.
    d_un_restrict_op = grid_geom->lookupCoarsenOperator(un_sol_var, "CONSERVATIVE_COARSEN");
    d_us_restrict_op = grid_geom->lookupCoarsenOperator(us_sol_var, "CONSERVATIVE_COARSEN");
    d_p_restrict_op = grid_geom->lookupCoarsenOperator(p_sol_var, "CONSERVATIVE_COARSEN");

    CoarsenAlgorithm<NDIM> coarsen_alg;
    coarsen_alg.registerCoarsen(d_un_scr_idx, un_sol_idx, d_un_restrict_op);
    coarsen_alg.registerCoarsen(d_us_scr_idx, us_sol_idx, d_us_restrict_op);
    coarsen_alg.registerCoarsen(d_p_scr_idx, p_sol_idx, d_p_restrict_op);

    d_restrict_scheds.resize(finest_ln - coarsest_ln);
    // Note don't create one for finest_ln because you can't restrict to the finest level.
    for (int dst_ln = coarsest_ln; dst_ln < finest_ln; ++dst_ln)
    {
        d_restrict_scheds[dst_ln] =
            coarsen_alg.createSchedule(d_hierarchy->getPatchLevel(dst_ln), d_hierarchy->getPatchLevel(dst_ln + 1));
    }

    // Create operators for only filling ghost cels.
    RefineAlgorithm<NDIM> ghostfill_alg;
    ghostfill_alg.registerRefine(d_un_scr_idx, un_sol_idx, d_un_scr_idx, Pointer<RefineOperator<NDIM>>());
    ghostfill_alg.registerRefine(d_us_scr_idx, us_sol_idx, d_us_scr_idx, Pointer<RefineOperator<NDIM>>());
    ghostfill_alg.registerRefine(d_p_scr_idx, p_sol_idx, d_p_scr_idx, Pointer<RefineOperator<NDIM>>());
    d_ghostfill_no_restrict_scheds.resize(finest_ln - coarsest_ln + 1);
    for (int dst_ln = coarsest_ln; dst_ln <= finest_ln; ++dst_ln)
    {
        // We only want to fill in ghost cells from the current level
        // TODO: the second argument here should fill in physical boundary conditions. This only works for periodic
        // conditions.
        d_ghostfill_no_restrict_scheds[dst_ln] =
            ghostfill_alg.createSchedule(d_hierarchy->getPatchLevel(dst_ln), nullptr);
    }

    // Coarse-fine boundary operators
    d_sc_bdry_op = new CartSideDoubleQuadraticCFInterpolation();
    d_sc_bdry_op->setConsistentInterpolationScheme(false);
    d_sc_bdry_op->setPatchHierarchy(d_hierarchy);
    d_cc_bdry_op = new CartCellDoubleQuadraticCFInterpolation();
    d_cc_bdry_op->setConsistentInterpolationScheme(false);
    d_cc_bdry_op->setPatchHierarchy(d_hierarchy);

    Pointer<SideSynchCopyFillPattern> synch_pattern = new SideSynchCopyFillPattern();
    RefineAlgorithm<NDIM> synch_alg;
    synch_alg.registerRefine(d_un_scr_idx, un_sol_idx, d_un_scr_idx, Pointer<RefineOperator<NDIM>>(), synch_pattern);
    synch_alg.registerRefine(d_us_scr_idx, us_sol_idx, d_us_scr_idx, Pointer<RefineOperator<NDIM>>(), synch_pattern);
    d_synch_scheds.resize(finest_ln - coarsest_ln + 1);
    for (int dst_ln = coarsest_ln; dst_ln <= finest_ln; ++dst_ln)
    {
        d_synch_scheds[dst_ln] = synch_alg.createSchedule(d_hierarchy->getPatchLevel(dst_ln));
    }

    // Determine overlap information
    // Get overlap information for setting patch boundary conditions.
    // Get overlap information for setting patch boundary conditions.
    d_patch_side_bc_box_overlap.resize(finest_ln + 1);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        const int num_local_patches = level->getProcessorMapping().getLocalIndices().getSize();
        d_patch_side_bc_box_overlap[ln].resize(num_local_patches);
        int patch_counter = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_counter)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& patch_box = patch->getBox();
            for (unsigned int axis = 0; axis < NDIM; ++axis)
            {
                const Box<NDIM> side_box = SideGeometry<NDIM>::toSideBox(patch_box, axis);
                const Box<NDIM> side_ghost_box = Box<NDIM>::grow(side_box, 1);
                d_patch_side_bc_box_overlap[ln][patch_counter][axis] = BoxList<NDIM>(side_ghost_box);
                d_patch_side_bc_box_overlap[ln][patch_counter][axis].removeIntersections(side_box);
            }
        }
    }

    d_patch_cell_bc_box_overlap.resize(finest_ln + 1);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);

        const int num_local_patches = level->getProcessorMapping().getLocalIndices().getSize();
        d_patch_cell_bc_box_overlap[ln].resize(num_local_patches);

        int patch_counter = 0;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++, ++patch_counter)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& patch_box = patch->getBox();
            const Box<NDIM>& ghost_box = Box<NDIM>::grow(patch_box, 1);

            d_patch_cell_bc_box_overlap[ln][patch_counter] = BoxList<NDIM>(ghost_box);
            d_patch_cell_bc_box_overlap[ln][patch_counter].removeIntersections(patch_box);
        }
    }

    // Get overlap information for re-setting patch boundary conditions during
    // smoothing.
    d_patch_side_neighbor_overlap.resize(finest_ln + 1);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        const int num_local_patches = level->getProcessorMapping().getLocalIndices().getSize();
        d_patch_side_neighbor_overlap[ln].resize(num_local_patches);
        int patch_counter1 = 0;
        for (PatchLevel<NDIM>::Iterator p1(level); p1; p1++, ++patch_counter1)
        {
            for (unsigned int axis = 0; axis < NDIM; ++axis)
            {
                d_patch_side_neighbor_overlap[ln][patch_counter1][axis].clear();
            }
            Pointer<Patch<NDIM>> dst_patch = level->getPatch(p1());
            const Box<NDIM>& dst_patch_box = dst_patch->getBox();
            const Box<NDIM>& dst_ghost_box = Box<NDIM>::grow(dst_patch_box, 1);
            int patch_counter2 = 0;
            for (PatchLevel<NDIM>::Iterator p2(level); patch_counter2 < patch_counter1; p2++, ++patch_counter2)
            {
                Pointer<Patch<NDIM>> src_patch = level->getPatch(p2());
                const Box<NDIM>& src_patch_box = src_patch->getBox();
                for (unsigned int axis = 0; axis < NDIM; ++axis)
                {
                    const Box<NDIM> overlap = SideGeometry<NDIM>::toSideBox(dst_ghost_box, axis) *
                                              SideGeometry<NDIM>::toSideBox(src_patch_box, axis);
                    if (!overlap.empty())
                    {
                        d_patch_side_neighbor_overlap[ln][patch_counter1][axis][p2()] = overlap;
                    }
                }
            }
        }
    }

    d_patch_cell_neighbor_overlap.resize(finest_ln + 1);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        const int num_local_patches = level->getProcessorMapping().getLocalIndices().getSize();
        d_patch_cell_neighbor_overlap[ln].resize(num_local_patches);
        int patch_counter1 = 0;
        for (PatchLevel<NDIM>::Iterator p1(level); p1; p1++, ++patch_counter1)
        {
            d_patch_cell_neighbor_overlap[ln][patch_counter1].clear();
            Pointer<Patch<NDIM>> dst_patch = level->getPatch(p1());
            const Box<NDIM>& dst_patch_box = dst_patch->getBox();
            const Box<NDIM>& dst_ghost_box = Box<NDIM>::grow(dst_patch_box, 1);
            int patch_counter2 = 0;
            for (PatchLevel<NDIM>::Iterator p2(level); patch_counter2 < patch_counter1; p2++, ++patch_counter2)
            {
                Pointer<Patch<NDIM>> src_patch = level->getPatch(p2());
                const Box<NDIM>& src_patch_box = src_patch->getBox();
                const Box<NDIM> overlap = dst_ghost_box * src_patch_box;
                if (!overlap.empty())
                {
                    d_patch_cell_neighbor_overlap[ln][patch_counter1][p2()] = overlap;
                }
            }
        }
    }

    d_is_initialized = true;
    return;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::deallocateOperatorState()
{
    // Delete the operators and schedules
    d_un_restrict_op.setNull();
    d_us_restrict_op.setNull();
    d_p_restrict_op.setNull();
    d_restrict_scheds.resize(0);

    d_un_prolong_op.setNull();
    d_us_prolong_op.setNull();
    d_p_prolong_op.setNull();
    d_prolong_scheds.resize(0);

    // Deallocate any data associated with the operator
    int coarsest_ln = 0;
    int finest_ln = d_hierarchy->getFinestLevelNumber();
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->deallocatePatchData(d_un_scr_idx);
        level->deallocatePatchData(d_us_scr_idx);
        level->deallocatePatchData(d_p_scr_idx);
        level->deallocatePatchData(d_scratch_idxs[0]);
        level->deallocatePatchData(d_scratch_idxs[1]);
        level->deallocatePatchData(d_scratch_idxs[2]);
    }

    d_is_initialized = false;
    return;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::setUnderRelaxationParamater(const double w)
{
    d_w = w;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::setCandDCoefficients(const double C, const double D)
{
    d_C = C;
    d_D = D;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::setViscosityCoefficient(const double eta_n, const double eta_s)
{
    d_eta_n = eta_n;
    d_eta_s = eta_s;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::setDragCoefficient(const double xi,
                                                                      const double nu_n,
                                                                      const double nu_s)
{
    d_xi = xi;
    d_nu_n = nu_n;
    d_nu_s = nu_s;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::performProlongation(const std::array<int, 3>& dst_idxs,
                                                                       const std::array<int, 3>& src_idxs,
                                                                       const int dst_ln)
{
    RefineAlgorithm<NDIM> refine_alg;
    refine_alg.registerRefine(dst_idxs[0], src_idxs[0], dst_idxs[0], d_un_prolong_op);
    refine_alg.registerRefine(dst_idxs[1], src_idxs[1], dst_idxs[1], d_us_prolong_op);
    refine_alg.registerRefine(dst_idxs[2], src_idxs[2], dst_idxs[2], d_p_prolong_op);

    refine_alg.resetSchedule(d_prolong_scheds[dst_ln]);
    d_prolong_scheds[dst_ln]->fillData(d_new_time);
    return;
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::performRestriction(const std::array<int, 3>& dst_idxs,
                                                                      const std::array<int, 3>& src_idxs,
                                                                      const int dst_ln)
{
    CoarsenAlgorithm<NDIM> coarsen_alg;
    coarsen_alg.registerCoarsen(dst_idxs[0], src_idxs[0], d_un_restrict_op);
    coarsen_alg.registerCoarsen(dst_idxs[1], src_idxs[1], d_us_restrict_op);
    coarsen_alg.registerCoarsen(dst_idxs[2], src_idxs[2], d_p_restrict_op);

    coarsen_alg.resetSchedule(d_restrict_scheds[dst_ln]);
    d_restrict_scheds[dst_ln]->coarsenData();
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::performGhostFilling(const std::array<int, 3>& dst_idxs,
                                                                       const int dst_ln)
{
    RefineAlgorithm<NDIM> refine_alg;
    refine_alg.registerRefine(dst_idxs[0], dst_idxs[0], dst_idxs[0], Pointer<RefineOperator<NDIM>>());
    refine_alg.registerRefine(dst_idxs[1], dst_idxs[1], dst_idxs[1], Pointer<RefineOperator<NDIM>>());
    refine_alg.registerRefine(dst_idxs[2], dst_idxs[2], dst_idxs[2], Pointer<RefineOperator<NDIM>>());

    refine_alg.resetSchedule(d_ghostfill_no_restrict_scheds[dst_ln]);
    d_ghostfill_no_restrict_scheds[dst_ln]->fillData(d_new_time);
}

void
VCTwoFluidStaggeredStokesBoxRelaxationFACOperator::performSynch(const std::array<int, 2>& dst_idxs, const int dst_ln)
{
    RefineAlgorithm<NDIM> refine_alg;
    refine_alg.registerRefine(dst_idxs[0], dst_idxs[0], dst_idxs[0], Pointer<RefineOperator<NDIM>>());
    refine_alg.registerRefine(dst_idxs[1], dst_idxs[1], dst_idxs[1], Pointer<RefineOperator<NDIM>>());

    refine_alg.resetSchedule(d_synch_scheds[dst_ln]);
    d_synch_scheds[dst_ln]->fillData(d_new_time);
}

/////////////////////////////// PRIVATE //////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
} // namespace IBTK

//////////////////////////////////////////////////////////////////////////////
