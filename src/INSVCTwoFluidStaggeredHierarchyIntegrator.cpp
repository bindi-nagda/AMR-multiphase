// ---------------------------------------------------------------------
//
// Copyright (c) 2014 - 2022 by the IBAMR developers
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

#include "ibamr/AdvDiffHierarchyIntegrator.h"
#include "ibamr/ConvectiveOperator.h"
#include "ibamr/INSHierarchyIntegrator.h"
#include "ibamr/INSIntermediateVelocityBcCoef.h"
#include "ibamr/INSProjectionBcCoef.h"
#include "ibamr/INSStaggeredConvectiveOperatorManager.h"
#include "ibamr/INSStaggeredPressureBcCoef.h"
#include "ibamr/INSStaggeredVelocityBcCoef.h"
#include "ibamr/StaggeredStokesBlockPreconditioner.h"
#include "ibamr/StaggeredStokesFACPreconditioner.h"
#include "ibamr/StaggeredStokesPhysicalBoundaryHelper.h"
#include "ibamr/StaggeredStokesSolver.h"
#include "ibamr/StaggeredStokesSolverManager.h"
#include "ibamr/StokesSpecifications.h"
#include "ibamr/ibamr_enums.h"
#include "ibamr/ibamr_utilities.h"
#include "ibamr/namespaces.h" // IWYU pragma: keep

#include "ibtk/CCPoissonSolverManager.h"
#include "ibtk/CartCellDoubleBoundsPreservingConservativeLinearRefine.h"
#include "ibtk/CartGridFunction.h"
#include "ibtk/CartSideDoubleDivPreservingRefine.h"
#include "ibtk/CartSideDoubleRT0Refine.h"
#include "ibtk/CartSideDoubleSpecializedLinearRefine.h"
#include "ibtk/CartSideRobinPhysBdryOp.h"
#include "ibtk/HierarchyGhostCellInterpolation.h"
#include "ibtk/HierarchyIntegrator.h"
#include "ibtk/HierarchyMathOps.h"
#include "ibtk/IBTK_MPI.h"
#include "ibtk/KrylovLinearSolver.h"
#include "ibtk/LinearSolver.h"
#include "ibtk/NewtonKrylovSolver.h"
#include "ibtk/PoissonSolver.h"
#include "ibtk/SCPoissonSolverManager.h"
#include "ibtk/SideDataSynchronization.h"
#include "ibtk/ibtk_enums.h"
#include "ibtk/ibtk_utilities.h"
#include <ibtk/CartGridFunctionSet.h>
#include <ibtk/PETScKrylovLinearSolver.h>
#include <ibtk/PETScKrylovPoissonSolver.h>

#include "ArrayData.h"
#include "BasePatchHierarchy.h"
#include "BasePatchLevel.h"
#include "Box.h"
#include "CartesianGridGeometry.h"
#include "CartesianPatchGeometry.h"
#include "CellData.h"
#include "CellIndex.h"
#include "CellIterator.h"
#include "CellVariable.h"
#include "CoarsenAlgorithm.h"
#include "CoarsenOperator.h"
#include "CoarsenSchedule.h"
#include "ComponentSelector.h"
#include "FaceData.h"
#include "FaceVariable.h"
#include "GriddingAlgorithm.h"
#include "HierarchyCellDataOpsReal.h"
#include "HierarchyDataOpsManager.h"
#include "HierarchyDataOpsReal.h"
#include "HierarchyFaceDataOpsReal.h"
#include "HierarchySideDataOpsReal.h"
#include "Index.h"
#include "IntVector.h"
#include "LocationIndexRobinBcCoefs.h"
#include "MultiblockDataTranslator.h"
#include "Patch.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "PatchSideDataOpsReal.h"
#include "PoissonSpecifications.h"
#include "RefineAlgorithm.h"
#include "RefineOperator.h"
#include "RefinePatchStrategy.h"
#include "RefineSchedule.h"
#include "RobinBcCoefStrategy.h"
#include "SAMRAIVectorReal.h"
#include "SideData.h"
#include "SideIndex.h"
#include "SideVariable.h"
#include "Variable.h"
#include "VariableContext.h"
#include "VariableDatabase.h"
#include "VisItDataWriter.h"
#include "tbox/Array.h"
#include "tbox/Database.h"
#include "tbox/MathUtilities.h"
#include "tbox/MemoryDatabase.h"
#include "tbox/PIO.h"
#include "tbox/Pointer.h"
#include "tbox/Utilities.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// Local includes
#include "FullFACPreconditioner.h"
#include "INSVCTwoFluidStaggeredHierarchyIntegrator.h"
#include "VCTwoFluidStaggeredStokesBoxRelaxationFACOperator.h"
#include "VCTwoFluidStaggeredStokesOperator.h"
#include "utility_functions.h"

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace IBAMR
{

/////////////////////////////// PUBLIC ///////////////////////////////////////
INSVCTwoFluidStaggeredHierarchyIntegrator::INSVCTwoFluidStaggeredHierarchyIntegrator(
    std::string object_name,
    Pointer<Database> input_db,
    Pointer<CartesianGridGeometry<NDIM>> grid_geometry,
    bool register_for_restart)
    : INSHierarchyIntegrator(std::move(object_name),
                             input_db,
                             new SideVariable<NDIM, double>(object_name + "::Us"),
                             "CONSERVATIVE_COARSEN",
                             "CONSERVATIVE_LINEAR_REFINE",
                             new CellVariable<NDIM, double>(object_name + "::P"),
                             "CONSERVATIVE_COARSEN",
                             "LINEAR_REFINE",
                             new SideVariable<NDIM, double>(object_name + "::F"),
                             "CONSERVATIVE_COARSEN",
                             "CONSERVATIVE_LINEAR_REFINE",
                             new CellVariable<NDIM, double>(object_name + "::Q"),
                             "CONSERVATIVE_COARSEN",
                             "CONSTANT_REFINE",
                             register_for_restart),
      d_regrid_writer(new VisItDataWriter<NDIM>("RegridWriter", "regrid_viz"))
{
    if (input_db->keyExists("viscous_time_stepping_type"))
        d_viscous_time_stepping_type =
            string_to_enum<TimeSteppingType>(input_db->getString("viscous_time_stepping_type"));
    if (input_db->keyExists("rho")) d_rho = input_db->getDouble("rho");
    if (input_db->keyExists("solver_db")) d_solver_db = input_db->getDatabase("solver_db");
    if (input_db->keyExists("precond_db")) d_precond_db = input_db->getDatabase("precond_db");
    if (input_db->keyExists("w")) d_w = input_db->getDouble("w");
    if (input_db->keyExists("use_preconditioner")) d_use_preconditioner = input_db->getBool("use_preconditioner");
    if (input_db->keyExists("use_grad_tagging")) d_use_grad_tagging = input_db->getBool("use_grad_tagging");
    if (input_db->keyExists("grad_rel_thresh")) input_db->getArray("grad_rel_thresh", d_rel_grad_thresh);
    if (input_db->keyExists("grad_abs_thresh")) input_db->getArray("grad_abs_thresh", d_abs_grad_thresh);
    if (input_db->keyExists("make_div_rhs_sum_to_zero"))
        d_make_div_rhs_sum_to_zero = input_db->getBool("make_div_rhs_sum_to_zero");
    if (input_db->keyExists("regrid_solve_db")) d_regrid_solve_db = input_db->getDatabase("regrid_solve_db");
    if (input_db->keyExists("regrid_solve_pc_db")) d_regrid_solve_pc_db = input_db->getDatabase("regrid_solve_pc_db");
    d_un_sc_var = new SideVariable<NDIM, double>(d_object_name + "::un_sc");
    d_us_sc_var = new SideVariable<NDIM, double>(d_object_name + "::us_sc");

    // As of the moment, we DO NOT account for advection in the momentum equation. To account for this in the number of
    // cycles, we set d_creeping_flow = true.
    d_creeping_flow = true;

    // Make sure viscous time stepping type is valid for this class
    if (d_viscous_time_stepping_type != TimeSteppingType::BACKWARD_EULER &&
        d_viscous_time_stepping_type != TimeSteppingType::TRAPEZOIDAL_RULE)
        TBOX_ERROR(d_object_name + ": Viscous time step type " +
                   enum_to_string<TimeSteppingType>(d_viscous_time_stepping_type) +
                   " not valid. Must use BACKWARD_EULER or TRAPEZOIDAL_RULE");
    return;
} // INSVCTwoFluidStaggeredHierarchyIntegrator

INSVCTwoFluidStaggeredHierarchyIntegrator::~INSVCTwoFluidStaggeredHierarchyIntegrator()
{
    // intentionally blank
} // ~INSVCTwoFluidStaggeredHierarchyIntegrator

Pointer<ConvectiveOperator>
INSVCTwoFluidStaggeredHierarchyIntegrator::getConvectiveOperator()
{
    return nullptr;
} // getConvectiveOperator

Pointer<PoissonSolver>
INSVCTwoFluidStaggeredHierarchyIntegrator::getVelocitySubdomainSolver()
{
    return nullptr;
} // getVelocitySubdomainSolver

Pointer<PoissonSolver>
INSVCTwoFluidStaggeredHierarchyIntegrator::getPressureSubdomainSolver()
{
    return nullptr;
} // getPressureSubdomainSolver

Pointer<SideVariable<NDIM, double>>
INSVCTwoFluidStaggeredHierarchyIntegrator::getSolventVariable() const
{
    return d_us_sc_var;
}

Pointer<SideVariable<NDIM, double>>
INSVCTwoFluidStaggeredHierarchyIntegrator::getNetworkVariable() const
{
    return d_un_sc_var;
}

Pointer<CellVariable<NDIM, double>>
INSVCTwoFluidStaggeredHierarchyIntegrator::getPressureVariable() const
{
    return d_P_var;
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::setViscosityCoefficient(const double eta_n, const double eta_s)
{
    d_eta_n = eta_n;
    d_eta_s = eta_s;
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::setDragCoefficient(const double xi, const double nu_n, const double nu_s)
{
    d_xi = xi;
    d_nu_n = nu_n;
    d_nu_s = nu_s;
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::setInitialData(Pointer<CartGridFunction> un_fcn,
                                                          Pointer<CartGridFunction> us_fcn,
                                                          Pointer<CartGridFunction> p_fcn)
{
    d_un_init_fcn = un_fcn;
    d_us_init_fcn = us_fcn;
    d_p_init_fcn = p_fcn;
    return;
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::setForcingFunctions(Pointer<CartGridFunction> fn_fcn,
                                                               Pointer<CartGridFunction> fs_fcn,
                                                               Pointer<CartGridFunction> fp_fcn)
{
    if (fn_fcn)
    {
        if (d_f_un_fcn)
        {
            Pointer<CartGridFunctionSet> p_f_un_fcn = d_f_un_fcn;
            if (p_f_un_fcn)
            {
                p_f_un_fcn->addFunction(fn_fcn);
            }
            else
            {
                p_f_un_fcn = new CartGridFunctionSet(d_object_name + "::fn_fcn_set");
                p_f_un_fcn->addFunction(d_f_un_fcn);
                p_f_un_fcn->addFunction(fn_fcn);
                d_f_un_fcn = p_f_un_fcn;
            }
        }
        else
        {
            d_f_un_fcn = fn_fcn;
        }
    }

    if (fs_fcn)
    {
        if (d_f_us_fcn)
        {
            Pointer<CartGridFunctionSet> p_f_us_fcn = d_f_us_fcn;
            if (p_f_us_fcn)
            {
                p_f_us_fcn->addFunction(fs_fcn);
            }
            else
            {
                p_f_us_fcn = new CartGridFunctionSet(d_object_name + "::fs_fcn_set");
                p_f_us_fcn->addFunction(d_f_us_fcn);
                p_f_us_fcn->addFunction(fs_fcn);
                d_f_us_fcn = p_f_us_fcn;
            }
        }
        else
        {
            d_f_us_fcn = fs_fcn;
        }
    }

    if (fp_fcn)
    {
        if (d_f_p_fcn)
        {
            Pointer<CartGridFunctionSet> p_f_p_fcn = d_f_p_fcn;
            if (p_f_p_fcn)
            {
                p_f_p_fcn->addFunction(fp_fcn);
            }
            else
            {
                p_f_p_fcn = new CartGridFunctionSet(d_object_name + "::fp_fcn_set");
                p_f_p_fcn->addFunction(d_f_p_fcn);
                p_f_p_fcn->addFunction(fp_fcn);
                d_f_p_fcn = p_f_p_fcn;
            }
        }
        else
        {
            d_f_p_fcn = fp_fcn;
        }
    }
    return;
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::setForcingFunctionsScaled(Pointer<CartGridFunction> fn_fcn,
                                                                     Pointer<CartGridFunction> fs_fcn)
{
    if (fn_fcn)
    {
        if (d_f_un_thn_fcn)
        {
            Pointer<CartGridFunctionSet> p_f_un_fcn = d_f_un_thn_fcn;
            if (p_f_un_fcn)
            {
                p_f_un_fcn->addFunction(fn_fcn);
            }
            else
            {
                p_f_un_fcn = new CartGridFunctionSet(d_object_name + "::fn_thn_fcn_set");
                p_f_un_fcn->addFunction(d_f_un_thn_fcn);
                p_f_un_fcn->addFunction(fn_fcn);
                d_f_un_thn_fcn = p_f_un_fcn;
            }
        }
        else
        {
            d_f_un_thn_fcn = fn_fcn;
        }
    }

    if (fs_fcn)
    {
        if (d_f_us_ths_fcn)
        {
            Pointer<CartGridFunctionSet> p_f_us_fcn = d_f_us_ths_fcn;
            if (p_f_us_fcn)
            {
                p_f_us_fcn->addFunction(fs_fcn);
            }
            else
            {
                p_f_us_fcn = new CartGridFunctionSet(d_object_name + "::fs_ths_fcn_set");
                p_f_us_fcn->addFunction(d_f_us_ths_fcn);
                p_f_us_fcn->addFunction(fs_fcn);
                d_f_us_ths_fcn = p_f_us_fcn;
            }
        }
        else
        {
            d_f_us_ths_fcn = fs_fcn;
        }
    }
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::setInitialNetworkVolumeFraction(Pointer<CartGridFunction> thn_init_fcn)
{
    d_thn_init_fcn = thn_init_fcn;
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::setNetworkVolumeFractionFunction(Pointer<CartGridFunction> thn_fcn,
                                                                            bool use_as_initial_data)
{
    // Make sure we have a valid pointer.
    TBOX_ASSERT(thn_fcn);
    d_thn_fcn = thn_fcn;
    if (use_as_initial_data) d_thn_init_fcn = thn_fcn;
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::advectNetworkVolumeFraction(
    Pointer<AdvDiffHierarchyIntegrator> adv_diff_integrator)
{
    registerAdvDiffHierarchyIntegrator(adv_diff_integrator);
    d_thn_integrator = adv_diff_integrator;

    // Set up thn to be advected
    d_thn_cc_var = new CellVariable<NDIM, double>(d_object_name + "::thn_cc");
    d_thn_integrator->registerTransportedQuantity(d_thn_cc_var, true /*output_Q*/);
    d_thn_integrator->setAdvectionVelocity(d_thn_cc_var, d_U_adv_diff_var);
    d_thn_integrator->setAdvectionVelocityIsDivergenceFree(d_U_adv_diff_var, false); // Not divergence free in general.
    d_thn_integrator->setInitialConditions(d_thn_cc_var, d_thn_init_fcn);
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::initializeHierarchyIntegrator(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                                                         Pointer<GriddingAlgorithm<NDIM>> gridding_alg)
{
    if (d_integrator_is_initialized) return;

    // Here we do all we need to ensure that calls to advanceHierarchy() or integrateHierarchy() are valid.
    // NOTE: This function is called before the patch hierarchy has valid patch levels.
    // To set initial data, we should do this in initializeLevelDataSpecialized().

    // First create the variables we need.
    // NOTE: d_P_var is a member variable of the base class.
    // NOTE: This may have been created above with the advection diffusion integrator.
    if (!d_thn_cc_var) d_thn_cc_var = new CellVariable<NDIM, double>(d_object_name + "::thn_cc");
    d_grad_thn_var = new CellVariable<NDIM, double>(d_object_name + "grad_thn_cc", NDIM);
    d_f_un_sc_var = new SideVariable<NDIM, double>(d_object_name + "::f_un_sc");
    d_f_us_sc_var = new SideVariable<NDIM, double>(d_object_name + "::f_us_sc");
    d_f_cc_var = new CellVariable<NDIM, double>(d_object_name + "::f_cc");
    d_un_rhs_var = new SideVariable<NDIM, double>(d_object_name + "::un_rhs");
    d_us_rhs_var = new SideVariable<NDIM, double>(d_object_name + "::us_rhs");
    d_p_rhs_var = new CellVariable<NDIM, double>(d_object_name + "::p_rhs");

    // Register variables with the integrator. Those with states (Velocities) have associated current, new, and scratch
    // indices. NOTE: Pressure is NOT a state variable, but we keep track of current and new values for initial guesses
    // for the solver.
    int un_cur_idx, un_scr_idx, un_new_idx;
    int us_cur_idx, us_scr_idx, us_new_idx;
    int p_cur_idx, p_scr_idx, p_new_idx;
    registerVariable(un_cur_idx,
                     un_new_idx,
                     un_scr_idx,
                     d_un_sc_var,
                     IntVector<NDIM>(1),
                     "CONSERVATIVE_COARSEN",
                     "CONSERVATIVE_LINEAR_REFINE",
                     d_un_init_fcn);
    registerVariable(us_cur_idx,
                     us_new_idx,
                     us_scr_idx,
                     d_us_sc_var,
                     IntVector<NDIM>(1),
                     "CONSERVATIVE_COARSEN",
                     "CONSERVATIVE_LINEAR_REFINE",
                     d_us_init_fcn);
    registerVariable(p_cur_idx,
                     p_new_idx,
                     p_scr_idx,
                     d_P_var,
                     IntVector<NDIM>(1),
                     "CONSERVATIVE_COARSEN",
                     "CONSERVATIVE_LINEAR_REFINE",
                     d_p_init_fcn);

    d_os_var = new OutersideVariable<NDIM, double>(d_object_name + "::os");
    registerVariable(d_os_idx, d_os_var, 0, getCurrentContext());

    // Everything else only gets a scratch context, which is deallocated at the end of each time step.
    // Note the forces need ghost cells for modifying the RHS to account for non-homogenous boundary conditions.
    int thn_cur_idx, thn_scr_idx, thn_new_idx, f_p_idx, f_un_idx, f_us_idx, un_rhs_idx, us_rhs_idx, p_rhs_idx;
    registerVariable(f_p_idx, d_f_cc_var, IntVector<NDIM>(1), getScratchContext());
    registerVariable(f_un_idx, d_f_un_sc_var, IntVector<NDIM>(1), getScratchContext());
    registerVariable(f_us_idx, d_f_us_sc_var, IntVector<NDIM>(1), getScratchContext());
    registerVariable(thn_cur_idx, d_thn_cc_var, IntVector<NDIM>(1), getCurrentContext());
    registerVariable(thn_scr_idx, d_thn_cc_var, IntVector<NDIM>(1), getScratchContext());
    registerVariable(thn_new_idx, d_thn_cc_var, IntVector<NDIM>(1), getNewContext());
    registerVariable(un_rhs_idx, d_un_rhs_var, IntVector<NDIM>(1), getCurrentContext());
    registerVariable(us_rhs_idx, d_us_rhs_var, IntVector<NDIM>(1), getCurrentContext());
    registerVariable(p_rhs_idx, d_p_rhs_var, IntVector<NDIM>(1), getScratchContext());

    // Register gradient of thn with the current context
    int grad_thn_idx;
    registerVariable(grad_thn_idx, d_grad_thn_var, IntVector<NDIM>(0), getCurrentContext());

    // Drawing variables. Velocities are node centered that get allocated with the current context.
    // Pressure is cell centered, so we can just use the current context for that.
    // Note: Using the current context for the velocities means drawing variables are allocated with the state
    // variables. This is important because the interface has no way of separately allocating and deallocating "drawing"
    // data.
    if (d_visit_writer)
    {
        d_un_draw_var = new NodeVariable<NDIM, double>(d_object_name + "::Un_draw", NDIM);
        d_us_draw_var = new NodeVariable<NDIM, double>(d_object_name + "::Us_draw", NDIM);
        d_div_draw_var = new CellVariable<NDIM, double>(d_object_name + "::Div_Draw");
        d_un_rhs_draw_var = new NodeVariable<NDIM, double>(d_object_name + "::Un_RHS_draw", NDIM);
        d_us_rhs_draw_var = new NodeVariable<NDIM, double>(d_object_name + "::Us_RHS_draw", NDIM);
        int un_draw_idx, us_draw_idx, div_draw_idx, un_rhs_draw_idx, us_rhs_draw_idx;
        registerVariable(un_draw_idx, d_un_draw_var, IntVector<NDIM>(0), getCurrentContext());
        registerVariable(us_draw_idx, d_us_draw_var, IntVector<NDIM>(0), getCurrentContext());
        registerVariable(div_draw_idx, d_div_draw_var, IntVector<NDIM>(1), getCurrentContext());
        registerVariable(un_rhs_draw_idx, d_un_rhs_draw_var, IntVector<NDIM>(0), getCurrentContext());
        registerVariable(us_rhs_draw_idx, d_us_rhs_draw_var, IntVector<NDIM>(0), getCurrentContext());

        d_visit_writer->registerPlotQuantity("Un", "VECTOR", un_draw_idx, 0, 1.0, "NODE");
        d_regrid_writer->registerPlotQuantity("Un", "VECTOR", un_draw_idx, 0, 1.0, "NODE");
        for (int d = 0; d < NDIM; ++d)
        {
            d_visit_writer->registerPlotQuantity("Un_" + std::to_string(d), "SCALAR", un_draw_idx, d, 1.0, "NODE");
            d_regrid_writer->registerPlotQuantity("Un_" + std::to_string(d), "SCALAR", un_draw_idx, d, 1.0, "NODE");
        }

        d_visit_writer->registerPlotQuantity("Us", "VECTOR", us_draw_idx, 0, 1.0, "NODE");
        d_regrid_writer->registerPlotQuantity("Us", "VECTOR", us_draw_idx, 0, 1.0, "NODE");
        for (int d = 0; d < NDIM; ++d)
        {
            d_visit_writer->registerPlotQuantity("Us_" + std::to_string(d), "SCALAR", us_draw_idx, d, 1.0, "NODE");
            d_regrid_writer->registerPlotQuantity("Us_" + std::to_string(d), "SCALAR", us_draw_idx, d, 1.0, "NODE");
        }

        d_visit_writer->registerPlotQuantity("P", "SCALAR", p_cur_idx, 0, 1.0, "CELL");
        d_regrid_writer->registerPlotQuantity("P", "SCALAR", p_cur_idx, 0, 1.0, "CELL");

        d_visit_writer->registerPlotQuantity("Div", "SCALAR", div_draw_idx, 0, 1.0, "CELL");
        d_regrid_writer->registerPlotQuantity("Div", "SCALAR", div_draw_idx, 0, 1.0, "CELL");

        d_visit_writer->registerPlotQuantity("rhs_Un", "VECTOR", un_rhs_draw_idx, 0, 1.0, "NODE");
        for (int d = 0; d < NDIM; ++d)
            d_visit_writer->registerPlotQuantity(
                "rhs_Un_" + std::to_string(d), "SCALAR", un_rhs_draw_idx, d, 1.0, "NODE");

        d_visit_writer->registerPlotQuantity("rhs_Us", "VECTOR", us_rhs_draw_idx, 0, 1.0, "NODE");
        for (int d = 0; d < NDIM; ++d)
            d_visit_writer->registerPlotQuantity(
                "rhs_Us_" + std::to_string(d), "SCALAR", us_rhs_draw_idx, d, 1.0, "NODE");

        // Only need to plot this variable if we aren't advecting it.
        // If we do advect theta, the advection integrator will plot it.
        if (d_thn_fcn) d_visit_writer->registerPlotQuantity("Thn", "SCALAR", thn_cur_idx, 0, 1.0, "CELL");
    }

    // Create the hierarchy data operations
    auto hier_ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
    d_hier_cc_data_ops =
        hier_ops_manager->getOperationsDouble(new CellVariable<NDIM, double>("cc_var"), hierarchy, true /*get_unique*/);
    d_hier_sc_data_ops =
        hier_ops_manager->getOperationsDouble(new SideVariable<NDIM, double>("sc_var"), hierarchy, true /*get_unique*/);
    d_hier_fc_data_ops =
        hier_ops_manager->getOperationsDouble(new FaceVariable<NDIM, double>("fc_var"), hierarchy, true /*get_unique*/);

    d_integrator_is_initialized = true;
    return;
} // initializeHierarchyIntegrator

void
INSVCTwoFluidStaggeredHierarchyIntegrator::initializeLevelDataSpecialized(Pointer<BasePatchHierarchy<NDIM>> hierarchy,
                                                                          int level_number,
                                                                          double init_data_time,
                                                                          bool can_be_refined,
                                                                          bool initial_time,
                                                                          Pointer<BasePatchLevel<NDIM>> old_level,
                                                                          bool allocate_data)
{
    // Do any kind of Hierarchy specific initialization on a given patch level.
    // Note: All initialization and regridding of state variables is managed by HierarchyIntegrator.
    // Example uses of this include mantaining a vorticity value to see where things should be refined or checking norms
    // of quantities before and after regridding.
    Pointer<PatchLevel<NDIM>> new_level = hierarchy->getPatchLevel(level_number);

    // Compute gradient of theta for initial tagging
    if (initial_time)
    {
        HierarchyDataOpsManager<NDIM>* hier_ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
        Pointer<HierarchyCellDataOpsReal<NDIM, double>> hier_cc_data_ops =
            hier_ops_manager->getOperationsDouble(d_grad_thn_var, d_hierarchy, true);

        auto var_db = VariableDatabase<NDIM>::getDatabase();
        const int grad_thn_idx = var_db->mapVariableAndContextToIndex(d_grad_thn_var, getCurrentContext());
        const int thn_idx = var_db->mapVariableAndContextToIndex(d_thn_cc_var, getCurrentContext());

        // Fill in with initial conditions.
        d_thn_init_fcn->setDataOnPatchLevel(thn_idx, d_thn_cc_var, new_level, init_data_time, initial_time);

        // Now fill in ghost cells
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comp(1);
        ghost_cell_comp[0] = ITC(thn_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "LINEAR", false, nullptr);

        Pointer<HierarchyGhostCellInterpolation> ghost_cell_fill = new HierarchyGhostCellInterpolation();
        // Note: when we create a new level, the coarser levels should already be created. So we can use that data to
        // fill ghost cells.
        ghost_cell_fill->initializeOperatorState(ghost_cell_comp, hierarchy, 0, level_number);
        HierarchyMathOps hier_math_ops("HierMathOps", hierarchy, 0, level_number);
        hier_math_ops.grad(grad_thn_idx, d_grad_thn_var, 1.0, thn_idx, d_thn_cc_var, ghost_cell_fill, init_data_time);
        // TODO: Replace this with a max over the L2 norm.
        d_max_grad_thn = hier_cc_data_ops->maxNorm(grad_thn_idx, IBTK::invalid_index);

        for (PatchLevel<NDIM>::Iterator p(new_level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = new_level->getPatch(p());
            Pointer<SideData<NDIM, double>> un_rhs_data = patch->getPatchData(d_un_rhs_var, getCurrentContext());
            Pointer<SideData<NDIM, double>> us_rhs_data = patch->getPatchData(d_us_rhs_var, getCurrentContext());
            un_rhs_data->fillAll(0.0);
            us_rhs_data->fillAll(0.0);
        }
    }
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::applyGradientDetectorSpecialized(
    const Pointer<BasePatchHierarchy<NDIM>> hierarchy,
    const int level_num,
    const double error_data_time,
    const int tag_idx,
    const bool initial_time,
    const bool uses_richardson_extrapolation_too)
{
    // Fill in the tag_idx with 1 if the cell index on a given level should be refined.
    // tag_idx corresponds to patch data CellData<NDIM, int>.
    // Note: d_grad_thn_var is filled during postprocessHierarchyIntegrator.
    if (d_use_grad_tagging)
    {
        // Determine thresholds for the level
        double grad_rel_thresh = std::numeric_limits<double>::max();
        double grad_abs_thresh = std::numeric_limits<double>::max();
        if (d_abs_grad_thresh.size() > 0)
            grad_abs_thresh = d_abs_grad_thresh[std::max(std::min(level_num, d_abs_grad_thresh.size() - 1), 0)];
        if (d_rel_grad_thresh.size() > 0)
            grad_rel_thresh = d_rel_grad_thresh[std::max(std::min(level_num, d_rel_grad_thresh.size() - 1), 0)];
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(level_num);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            Pointer<CellData<NDIM, double>> grad_thn_data = patch->getPatchData(d_grad_thn_var, getCurrentContext());
            Pointer<CellData<NDIM, int>> tagged_data = patch->getPatchData(tag_idx);

            for (CellIterator<NDIM> ci(patch->getBox()); ci; ci++) // cell-centers
            {
                const CellIndex<NDIM>& idx = ci();

                double grad_thn_norm = 0.0;
                for (int d = 0; d < NDIM; ++d) grad_thn_norm += (*grad_thn_data)(idx, d) * (*grad_thn_data)(idx, d);
                grad_thn_norm = std::sqrt(grad_thn_norm);

                if (grad_thn_norm > grad_abs_thresh || (grad_thn_norm / d_max_grad_thn) > grad_rel_thresh)
                    (*tagged_data)(idx) = 1;
            }
        }
    }
    return;
} // applyGradientDetectorSpecialized

void
INSVCTwoFluidStaggeredHierarchyIntegrator::resetHierarchyConfigurationSpecialized(
    SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
    int coarsest_level,
    int finest_level)
{
    // Do any kind of Hierarchy specific configuration on a new patch hierarchy.
    // If there are synchronization objects that need to be created, do that here.

    // Reset the hierarchy operations objects.
    d_hier_cc_data_ops->setPatchHierarchy(hierarchy);
    d_hier_cc_data_ops->resetLevels(0, hierarchy->getFinestLevelNumber());

    d_hier_sc_data_ops->setPatchHierarchy(hierarchy);
    d_hier_sc_data_ops->resetLevels(0, hierarchy->getFinestLevelNumber());

    d_hier_fc_data_ops->setPatchHierarchy(hierarchy);
    d_hier_fc_data_ops->resetLevels(0, hierarchy->getFinestLevelNumber());
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::preprocessIntegrateHierarchy(const double current_time,
                                                                        const double new_time,
                                                                        const int num_cycles)
{
    // Do anything that needs to be done before we call integrateHierarchy().
    INSHierarchyIntegrator::preprocessIntegrateHierarchy(current_time, new_time, num_cycles);

    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    const double dt = new_time - current_time;
    const double half_time = 0.5 * (new_time + current_time);

    // Pull out current solution components
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int un_cur_idx = var_db->mapVariableAndContextToIndex(d_un_sc_var, getCurrentContext());
    const int us_cur_idx = var_db->mapVariableAndContextToIndex(d_us_sc_var, getCurrentContext());
    const int p_cur_idx = var_db->mapVariableAndContextToIndex(d_P_var, getCurrentContext());
    const int un_new_idx = var_db->mapVariableAndContextToIndex(d_un_sc_var, getNewContext());
    const int us_new_idx = var_db->mapVariableAndContextToIndex(d_us_sc_var, getNewContext());
    const int p_new_idx = var_db->mapVariableAndContextToIndex(d_P_var, getNewContext());
    const int un_scr_idx = var_db->mapVariableAndContextToIndex(d_un_sc_var, getScratchContext());
    const int us_scr_idx = var_db->mapVariableAndContextToIndex(d_us_sc_var, getScratchContext());
    const int p_scr_idx = var_db->mapVariableAndContextToIndex(d_P_var, getScratchContext());

    // Allocate scratch and new data
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(d_scratch_data, current_time);
        level->allocatePatchData(d_new_data, new_time);
    }

    // Set initial guess
    d_hier_sc_data_ops->copyData(un_new_idx, un_cur_idx);
    d_hier_sc_data_ops->copyData(us_new_idx, us_cur_idx);
    d_hier_cc_data_ops->copyData(p_new_idx, p_cur_idx);
    d_hier_sc_data_ops->copyData(un_scr_idx, un_cur_idx);
    d_hier_sc_data_ops->copyData(us_scr_idx, us_cur_idx);
    d_hier_cc_data_ops->copyData(p_scr_idx, p_cur_idx);

    // Set up our solution vector
    d_sol_vec = new SAMRAIVectorReal<NDIM, double>(d_object_name + "::sol_vec", d_hierarchy, coarsest_ln, finest_ln);

    const int wgt_sc_idx = d_hier_math_ops->getSideWeightPatchDescriptorIndex();
    const int wgt_cc_idx = d_hier_math_ops->getCellWeightPatchDescriptorIndex();
    d_sol_vec->addComponent(d_un_sc_var, un_scr_idx, wgt_sc_idx, d_hier_sc_data_ops);
    d_sol_vec->addComponent(d_us_sc_var, us_scr_idx, wgt_sc_idx, d_hier_sc_data_ops);
    d_sol_vec->addComponent(d_P_var, p_scr_idx, wgt_cc_idx, d_hier_cc_data_ops);

    // Set up the RHS vector
    const int un_rhs_idx = var_db->mapVariableAndContextToIndex(d_un_rhs_var, getCurrentContext());
    const int us_rhs_idx = var_db->mapVariableAndContextToIndex(d_us_rhs_var, getCurrentContext());
    const int p_rhs_idx = var_db->mapVariableAndContextToIndex(d_p_rhs_var, getScratchContext());
    d_rhs_vec = new SAMRAIVectorReal<NDIM, double>(d_object_name + "::rhs_vec", d_hierarchy, coarsest_ln, finest_ln);
    d_rhs_vec->addComponent(d_un_rhs_var, un_rhs_idx, wgt_sc_idx, d_hier_sc_data_ops);
    d_rhs_vec->addComponent(d_us_rhs_var, us_rhs_idx, wgt_sc_idx, d_hier_sc_data_ops);
    d_rhs_vec->addComponent(d_p_rhs_var, p_rhs_idx, wgt_cc_idx, d_hier_cc_data_ops);
    d_rhs_vec->setToScalar(0.0);

    // Grab the theta index
    const int thn_cur_idx = var_db->mapVariableAndContextToIndex(d_thn_cc_var, getCurrentContext());
    const int thn_new_idx = var_db->mapVariableAndContextToIndex(d_thn_cc_var, getNewContext());
    const int thn_scr_idx = var_db->mapVariableAndContextToIndex(d_thn_cc_var, getScratchContext());
    if (d_thn_fcn)
    {
        // Set the correct data if applicable
        d_thn_fcn->setDataOnPatchHierarchy(
            thn_cur_idx, d_thn_cc_var, d_hierarchy, current_time, false, coarsest_ln, finest_ln);
        d_thn_fcn->setDataOnPatchHierarchy(
            thn_new_idx, d_thn_cc_var, d_hierarchy, new_time, false, coarsest_ln, finest_ln);
        d_hier_cc_data_ops->linearSum(thn_scr_idx, 0.5, thn_cur_idx, 0.5, thn_new_idx);
    }
    else
    {
        const int thn_adv_cur_idx =
            var_db->mapVariableAndContextToIndex(d_thn_cc_var, d_thn_integrator->getCurrentContext());
        // Otherwise set the new data to the best guess
        d_hier_cc_data_ops->copyData(thn_new_idx, thn_adv_cur_idx);
        d_hier_cc_data_ops->copyData(thn_cur_idx, thn_adv_cur_idx);
        d_hier_cc_data_ops->linearSum(thn_scr_idx, 0.5, thn_cur_idx, 0.5, thn_new_idx);
    }

    // Set up null vectors (if applicable)
    d_nul_vecs.resize(1);
    d_nul_vecs[0] = d_sol_vec->cloneVector("PressureNull"); // should delete the vector at the end
    d_nul_vecs[0]->allocateVectorData();
    d_nul_vecs[0]->setToScalar(0.0);
    // Pull out pressure component and set to constant
    for (int ln = 0; ln <= d_hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> p_data = d_nul_vecs[0]->getComponentPatchData(2, *patch);
            p_data->fillAll(1.0);
        }
    }

    // set-up RHS to treat viscosity and drag with backward Euler or Implicit Trapezoidal Rule:
    // RHS = f(n) + C*theta_i(n)*u_i(n) + D1*(viscous + drag) for  i = n, s
    double D1 = std::numeric_limits<double>::signaling_NaN();
    double D2 = std::numeric_limits<double>::signaling_NaN();
    const double C = d_rho / dt;

    switch (d_viscous_time_stepping_type)
    {
    case TRAPEZOIDAL_RULE:
        D1 = 0.5;
        D2 = -0.5;
        break;

    case BACKWARD_EULER:
        D1 = 0.0;
        D2 = -1.0;
        break;

    default:
        TBOX_ERROR("Unknown time stepping type " + enum_to_string<TimeSteppingType>(d_viscous_time_stepping_type) +
                   ". Valid options are BACKWARD_EULER and TRAPEZOIDAL_RULE.");
    }

    VCTwoFluidStaggeredStokesOperator RHS_op("RHS_op", true);
    // Divergence free condition and pressure are not time stepped. We do not need to account for the contributions in
    // the RHS.
    RHS_op.setCandDCoefficients(C, D1, 0.0, 0.0);
    RHS_op.setDragCoefficient(d_xi, d_nu_n, d_nu_s);
    RHS_op.setViscosityCoefficient(d_eta_n, d_eta_s);
    RHS_op.setThnIdx(thn_cur_idx); // Values at time t_n

    // Store results of applying stokes operator in rhs_vec
    RHS_op.initializeOperatorState(*d_sol_vec, *d_rhs_vec);
    RHS_op.apply(*d_sol_vec, *d_rhs_vec);

    // Set up the operators and solvers needed to solve the linear system.
    d_stokes_op = new VCTwoFluidStaggeredStokesOperator("stokes_op", true);
    d_stokes_op->setCandDCoefficients(C, D2);
    d_stokes_op->setDragCoefficient(d_xi, d_nu_n, d_nu_s);
    d_stokes_op->setViscosityCoefficient(d_eta_n, d_eta_s);
    d_stokes_op->setThnIdx(thn_new_idx); // Approximation at time t_{n+1}

    d_stokes_solver = new PETScKrylovLinearSolver("solver", d_solver_db, "solver_");
    d_stokes_solver->setOperator(d_stokes_op);

    // Now create a preconditioner
    if (d_use_preconditioner)
    {
        d_precond_op =
            new VCTwoFluidStaggeredStokesBoxRelaxationFACOperator("KrylovPrecondStrategy", "Krylov_precond_");
        d_precond_op->setThnIdx(thn_new_idx); // Approximation at time t_{n+1}
        d_precond_op->setDragCoefficient(d_xi, d_nu_n, d_nu_s);
        d_precond_op->setViscosityCoefficient(d_eta_n, d_eta_s);
        d_precond_op->setUnderRelaxationParamater(d_w);
        d_precond_op->setCandDCoefficients(C, D2);
        d_stokes_precond = new FullFACPreconditioner("KrylovPrecond", d_precond_op, d_precond_db, "Krylov_precond_");
        d_stokes_precond->setNullspace(false, d_nul_vecs);
        d_stokes_solver->setPreconditioner(d_stokes_precond);
    }

    d_stokes_solver->setNullspace(false, d_nul_vecs);
    d_stokes_solver->initializeSolverState(*d_sol_vec, *d_rhs_vec);

    // Set thn_cc_idx on the dense hierarchy.
    if (d_use_preconditioner)
    {
        Pointer<PatchHierarchy<NDIM>> dense_hierarchy = d_stokes_precond->getDenseHierarchy();
        if (d_thn_fcn)
        {
            // Allocate data
            for (int ln = 0; ln <= dense_hierarchy->getFinestLevelNumber(); ++ln)
            {
                Pointer<PatchLevel<NDIM>> level = dense_hierarchy->getPatchLevel(ln);
                if (!level->checkAllocated(thn_new_idx)) level->allocatePatchData(thn_new_idx, new_time);
            }
            d_thn_fcn->setDataOnPatchHierarchy(thn_new_idx,
                                               d_thn_cc_var,
                                               dense_hierarchy,
                                               new_time,
                                               false,
                                               0,
                                               dense_hierarchy->getFinestLevelNumber());
        }
        else
        {
            d_stokes_precond->transferToDense(thn_new_idx, true);
        }

        // Also fill in ghost cells on the dense hierarchy
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comp(1);
        ghost_cell_comp[0] = ITC(thn_new_idx,
                                 "CONSERVATIVE_LINEAR_REFINE",
                                 false,
                                 "NONE",
                                 "LINEAR",
                                 true,
                                 nullptr); // defaults to fill corner
        HierarchyGhostCellInterpolation ghost_cell_fill;
        ghost_cell_fill.initializeOperatorState(
            ghost_cell_comp, dense_hierarchy, 0, dense_hierarchy->getFinestLevelNumber());
        ghost_cell_fill.fillData(0.0);
    }

    // Set the forcing data if applicable.
    Pointer<SAMRAIVectorReal<NDIM, double>> f_vec = d_rhs_vec->cloneVector(d_object_name + "::F_vec");
    f_vec->allocateVectorData(current_time);
    f_vec->setToScalar(0.0);
    const int f_un_idx = var_db->mapVariableAndContextToIndex(d_f_un_sc_var, getScratchContext());
    const int f_us_idx = var_db->mapVariableAndContextToIndex(d_f_us_sc_var, getScratchContext());
    const int f_p_idx = var_db->mapVariableAndContextToIndex(d_f_cc_var, getScratchContext());
    if (d_f_un_fcn)
    {
        d_f_un_fcn->setDataOnPatchHierarchy(f_un_idx, d_f_un_sc_var, d_hierarchy, half_time);
        d_hier_sc_data_ops->add(f_vec->getComponentDescriptorIndex(0), f_un_idx, f_vec->getComponentDescriptorIndex(0));
    }
    if (d_f_us_fcn)
    {
        d_f_us_fcn->setDataOnPatchHierarchy(f_us_idx, d_f_us_sc_var, d_hierarchy, half_time);
        d_hier_sc_data_ops->add(f_vec->getComponentDescriptorIndex(1), f_us_idx, f_vec->getComponentDescriptorIndex(1));
    }
    // Divergence is not time-stepped. We need to prescribe the correct divergence at the time point for which we are
    // solving.
    if (d_f_p_fcn)
    {
        d_f_p_fcn->setDataOnPatchHierarchy(f_p_idx, d_f_cc_var, d_hierarchy, new_time);
        d_hier_cc_data_ops->add(f_vec->getComponentDescriptorIndex(2), f_p_idx, f_vec->getComponentDescriptorIndex(2));
    }

    // Now we account for theta variable forces
    // Fill in ghost cells for thn_scr_idx
    {
        // Also fill in ghost cells on the dense hierarchy
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_cell_comp(1);
        ghost_cell_comp[0] = ITC(thn_scr_idx,
                                 "CONSERVATIVE_LINEAR_REFINE",
                                 false,
                                 "NONE",
                                 "LINEAR",
                                 true,
                                 nullptr); // defaults to fill corner
        HierarchyGhostCellInterpolation ghost_cell_fill;
        ghost_cell_fill.initializeOperatorState(ghost_cell_comp, d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        ghost_cell_fill.fillData(half_time);
    }
    if (d_f_un_thn_fcn)
    {
        d_f_un_thn_fcn->setDataOnPatchHierarchy(f_un_idx, d_f_un_sc_var, d_hierarchy, half_time);
        multiply_sc_and_thn(f_un_idx, f_un_idx, thn_scr_idx, d_hierarchy);
        d_hier_sc_data_ops->add(f_vec->getComponentDescriptorIndex(0), f_un_idx, f_vec->getComponentDescriptorIndex(0));
    }
    if (d_f_us_ths_fcn)
    {
        d_f_us_ths_fcn->setDataOnPatchHierarchy(f_us_idx, d_f_us_sc_var, d_hierarchy, half_time);
        multiply_sc_and_ths(f_us_idx, f_us_idx, thn_scr_idx, d_hierarchy);
        d_hier_sc_data_ops->add(f_vec->getComponentDescriptorIndex(1), f_us_idx, f_vec->getComponentDescriptorIndex(1));
    }

    d_rhs_vec->add(d_rhs_vec, f_vec);
    f_vec->deallocateVectorData();
    f_vec->freeVectorComponents();

    // Set up the advection diffusion integrator
    for (const auto& adv_diff_integrator : d_adv_diff_hier_integrators)
    {
        const int adv_diff_num_cycles = adv_diff_integrator->getNumberOfCycles();
        const int U_adv_diff_cur_idx =
            var_db->mapVariableAndContextToIndex(d_U_adv_diff_var, adv_diff_integrator->getCurrentContext());
        const int U_adv_diff_scr_idx =
            var_db->mapVariableAndContextToIndex(d_U_adv_diff_var, adv_diff_integrator->getScratchContext());
        const int U_adv_diff_new_idx =
            var_db->mapVariableAndContextToIndex(d_U_adv_diff_var, adv_diff_integrator->getNewContext());
        if (isAllocatedPatchData(U_adv_diff_cur_idx)) copy_side_to_face(U_adv_diff_cur_idx, un_cur_idx, d_hierarchy);
        adv_diff_integrator->preprocessIntegrateHierarchy(current_time, new_time, adv_diff_num_cycles);
        if (isAllocatedPatchData(U_adv_diff_scr_idx))
            d_hier_fc_data_ops->copyData(U_adv_diff_scr_idx, U_adv_diff_cur_idx);
        if (isAllocatedPatchData(U_adv_diff_new_idx))
            d_hier_fc_data_ops->copyData(U_adv_diff_new_idx, U_adv_diff_cur_idx);
    }

    executePreprocessIntegrateHierarchyCallbackFcns(current_time, new_time, num_cycles);
    return;
} // preprocessIntegrateHierarchy

void
INSVCTwoFluidStaggeredHierarchyIntegrator::integrateHierarchy(const double current_time,
                                                              const double new_time,
                                                              const int cycle_num)
{
    INSHierarchyIntegrator::integrateHierarchy(current_time, new_time, cycle_num);
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int un_new_idx = var_db->mapVariableAndContextToIndex(d_un_sc_var, getNewContext());
    const int us_new_idx = var_db->mapVariableAndContextToIndex(d_us_sc_var, getNewContext());
    const int p_new_idx = var_db->mapVariableAndContextToIndex(d_P_var, getNewContext());
    // Update the state of the advection diffusion integrator
    for (const auto& adv_diff_integrator : d_adv_diff_hier_integrators)
        adv_diff_integrator->integrateHierarchy(current_time, new_time, cycle_num);

    // Update thn if necessary. We need to update it if we are using the preconditioner, which uses a different patch
    // hierarchy.
    if (d_thn_integrator)
    {
        const int thn_new_idx = var_db->mapVariableAndContextToIndex(d_thn_cc_var, getNewContext());
        const int thn_adv_new_idx =
            var_db->mapVariableAndContextToIndex(d_thn_cc_var, d_thn_integrator->getNewContext());
        d_hier_cc_data_ops->copyData(thn_new_idx, thn_adv_new_idx);
        if (d_use_preconditioner)
        {
            d_stokes_precond->transferToDense(thn_new_idx);
            // Also fill in ghost cells on the dense hierarchy
            Pointer<PatchHierarchy<NDIM>> dense_hierarchy = d_stokes_precond->getDenseHierarchy();
            using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
            std::vector<ITC> ghost_cell_comp(1);
            ghost_cell_comp[0] = ITC(thn_new_idx,
                                     "CONSERVATIVE_LINEAR_REFINE",
                                     false,
                                     "NONE",
                                     "LINEAR",
                                     true,
                                     nullptr); // defaults to fill corner
            HierarchyGhostCellInterpolation ghost_cell_fill;
            ghost_cell_fill.initializeOperatorState(
                ghost_cell_comp, dense_hierarchy, 0, dense_hierarchy->getFinestLevelNumber());
            ghost_cell_fill.fillData(0.0);
        }
    }
    const double dt = new_time - current_time;

    // Compute weighted sum for rhs divergence.
    // TODO: This needs a permanent home, rather than an if statement.
    if (d_make_div_rhs_sum_to_zero)
    {
        const int rhs_p_idx = d_rhs_vec->getComponentDescriptorIndex(2);
        HierarchyCellDataOpsReal<NDIM, double> hier_sc_data_ops(d_hierarchy, 0, d_hierarchy->getFinestLevelNumber());
        double integral = hier_sc_data_ops.integral(rhs_p_idx, d_hier_math_ops->getCellWeightPatchDescriptorIndex());
        plog << "Weighted integral = " << integral << "\n";
        hier_sc_data_ops.addScalar(rhs_p_idx, rhs_p_idx, -1.0 * integral);
        integral = hier_sc_data_ops.integral(rhs_p_idx, d_hier_math_ops->getCellWeightPatchDescriptorIndex());
        plog << "Weighted integral = " << integral << "\n";
    }

    // Set the initial guess for the system to be the most recent approximation to t^{n+1}
    d_hier_sc_data_ops->copyData(d_sol_vec->getComponentDescriptorIndex(0), un_new_idx);
    d_hier_sc_data_ops->copyData(d_sol_vec->getComponentDescriptorIndex(1), us_new_idx);
    d_hier_cc_data_ops->copyData(d_sol_vec->getComponentDescriptorIndex(2), p_new_idx);

    // Solve for un(n+1), us(n+1), p(n+1).
    d_sol_vec->setToScalar(0.0);
    bool converged = d_stokes_solver->solveSystem(*d_sol_vec, *d_rhs_vec);
    if (d_enable_logging)
        pout << "Stokes solver " << (converged ? "converged" : "failed to converge") << " after "
             << d_stokes_solver->getNumIterations() << " iterations\n";

    // Reset the solve vector to copy the "scratch" data into the "new" data
    d_hier_sc_data_ops->copyData(un_new_idx, d_sol_vec->getComponentDescriptorIndex(0));
    d_hier_sc_data_ops->copyData(us_new_idx, d_sol_vec->getComponentDescriptorIndex(1));
    d_hier_cc_data_ops->copyData(p_new_idx, d_sol_vec->getComponentDescriptorIndex(2));

    // Do any more updates to the advection diffusion variables
    for (const auto& adv_diff_hier_integrator : d_adv_diff_hier_integrators)
    {
        // Reset the velocities
        const int U_adv_diff_new_idx =
            var_db->mapVariableAndContextToIndex(d_U_adv_diff_var, adv_diff_hier_integrator->getNewContext());
        if (isAllocatedPatchData(U_adv_diff_new_idx)) copy_side_to_face(U_adv_diff_new_idx, un_new_idx, d_hierarchy);
        const int U_adv_diff_cur_idx =
            var_db->mapVariableAndContextToIndex(d_U_adv_diff_var, adv_diff_hier_integrator->getCurrentContext());
        const int U_adv_diff_scr_idx =
            var_db->mapVariableAndContextToIndex(d_U_adv_diff_var, adv_diff_hier_integrator->getScratchContext());
        if (isAllocatedPatchData(U_adv_diff_scr_idx))
            d_hier_fc_data_ops->linearSum(U_adv_diff_scr_idx, 0.5, U_adv_diff_new_idx, 0.5, U_adv_diff_cur_idx);

        // Now update the state variables. Note that cycle 0 has already been performed
        const int adv_diff_num_cycles = adv_diff_hier_integrator->getNumberOfCycles();
        if (d_current_num_cycles != adv_diff_num_cycles)
        {
            for (int adv_diff_cycle_num = 1; adv_diff_cycle_num < adv_diff_num_cycles; ++adv_diff_cycle_num)
                adv_diff_hier_integrator->integrateHierarchy(current_time, new_time, adv_diff_cycle_num);
        }
    }

    // Execuate any registered callbacks
    executeIntegrateHierarchyCallbackFcns(current_time, new_time, cycle_num);
    return;
} // integrateHierarchy

void
INSVCTwoFluidStaggeredHierarchyIntegrator::postprocessIntegrateHierarchy(const double current_time,
                                                                         const double new_time,
                                                                         const bool skip_synchronize_new_state_data,
                                                                         const int num_cycles)
{
    // Do anything that needs to be done after integrateHierarchy().
    INSHierarchyIntegrator::postprocessIntegrateHierarchy(
        current_time, new_time, skip_synchronize_new_state_data, num_cycles);

    // Calculate gradient of thn for grid cell tagging.
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int grad_thn_idx = var_db->mapVariableAndContextToIndex(d_grad_thn_var, getCurrentContext());
    const int thn_new_idx = var_db->mapVariableAndContextToIndex(d_thn_cc_var, getNewContext());
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_cell_comp(1);
    ghost_cell_comp[0] = ITC(thn_new_idx, "CONSERVATIVE_LINEAR_REFINE", false, "NONE", "LINEAR", false, nullptr);
    HierarchyGhostCellInterpolation hier_ghost_fill;
    hier_ghost_fill.initializeOperatorState(
        ghost_cell_comp, d_hierarchy, 0 /*coarsest_ln*/, d_hierarchy->getFinestLevelNumber());
    d_hier_math_ops->grad(grad_thn_idx,
                          d_grad_thn_var,
                          1.0 /*alpha*/,
                          thn_new_idx,
                          d_thn_cc_var,
                          Pointer<HierarchyGhostCellInterpolation>(&hier_ghost_fill, false),
                          new_time);
    // TODO: Replace this with a max over the L2 norm.
    d_max_grad_thn = d_hier_cc_data_ops->maxNorm(grad_thn_idx, IBTK::invalid_index);

    // Synchronize new state
    if (!skip_synchronize_new_state_data) synchronizeHierarchyData(NEW_DATA);

    // Note: Preconditioner should deallocate data as necessary.
    // Deallocate scratch data
    for (const auto& adv_diff_hier_integrator : d_adv_diff_hier_integrators)
    {
        const int adv_diff_num_cycles = adv_diff_hier_integrator->getNumberOfCycles();
        adv_diff_hier_integrator->postprocessIntegrateHierarchy(
            current_time, new_time, skip_synchronize_new_state_data, adv_diff_num_cycles);
    }

    for (auto& nul_vec : d_nul_vecs)
    {
        if (nul_vec)
        {
            nul_vec->deallocateVectorData();
            nul_vec->freeVectorComponents();
        }
    }
    d_nul_vecs.clear();

    // Execute any registered callbacks.
    executePostprocessIntegrateHierarchyCallbackFcns(
        current_time, new_time, skip_synchronize_new_state_data, num_cycles);

    d_stokes_op = nullptr;
    d_stokes_solver = nullptr;
    d_precond_op = nullptr;
    d_stokes_precond = nullptr;
    return;
} // postprocessIntegrateHierarchy

void
INSVCTwoFluidStaggeredHierarchyIntegrator::regridProjection()
{
    // During regridding, the coarsening and refining operations can introduce errors. Here, we project the velocity
    // onto a divergence free field.
    // TODO: Do we need to implement this? How do we project something that has a co-incompressibility condition.
    // TODO: Check divergence norms to ensure our regridding process does not introduce too large of errors.

    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();
    const int wgt_cc_idx = d_hier_math_ops->getCellWeightPatchDescriptorIndex();
    const double volume = d_hier_math_ops->getVolumeOfPhysicalDomain();

    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int p_scr_idx = var_db->mapVariableAndContextToIndex(d_P_var, getScratchContext());
    const int div_idx = var_db->mapVariableAndContextToIndex(d_div_draw_var, getCurrentContext());
    const int un_cur_idx = var_db->mapVariableAndContextToIndex(d_un_sc_var, getCurrentContext());
    const int un_scr_idx = var_db->mapVariableAndContextToIndex(d_un_sc_var, getScratchContext());
    const int us_cur_idx = var_db->mapVariableAndContextToIndex(d_us_sc_var, getCurrentContext());
    const int us_scr_idx = var_db->mapVariableAndContextToIndex(d_us_sc_var, getScratchContext());
    const int thn_cur_idx = var_db->mapVariableAndContextToIndex(d_thn_cc_var, getCurrentContext());

    // Setup solver vectors
    SAMRAIVectorReal<NDIM, double> sol_vec(d_object_name + "::sol_vec", d_hierarchy, coarsest_ln, finest_ln);
    sol_vec.addComponent(d_P_var, p_scr_idx, wgt_cc_idx, d_hier_cc_data_ops);
    SAMRAIVectorReal<NDIM, double> rhs_vec(d_object_name + "::rhs_vec", d_hierarchy, coarsest_ln, finest_ln);
    rhs_vec.addComponent(d_div_draw_var, div_idx, wgt_cc_idx, d_hier_cc_data_ops);

    Pointer<PoissonSolver> regrid_projection_solver =
        CCPoissonSolverManager::getManager()->allocateSolver(CCPoissonSolverManager::DEFAULT_KRYLOV_SOLVER,
                                                             "RegridSolver",
                                                             d_regrid_solve_db,
                                                             "regrid_solver_",
                                                             CCPoissonSolverManager::DEFAULT_FAC_PRECONDITIONER,
                                                             "RegridPC",
                                                             d_regrid_solve_pc_db,
                                                             "regrid_solver_pc_");
    PoissonSpecifications regrid_projection_spec(d_object_name + "::regrid_projection_spec");
    regrid_projection_spec.setCZero();
    regrid_projection_spec.setDConstant(1.0);
    LocationIndexRobinBcCoefs<NDIM> Phi_bc_coef;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
        Phi_bc_coef.setBoundarySlope(2 * d, 0.0);
        Phi_bc_coef.setBoundarySlope(2 * d + 1, 0.0);
    }

    regrid_projection_solver->setPoissonSpecifications(regrid_projection_spec);
    regrid_projection_solver->setPhysicalBcCoef(&Phi_bc_coef);
    regrid_projection_solver->setHomogeneousBc(true);
    regrid_projection_solver->setSolutionTime(d_integrator_time);
    regrid_projection_solver->setTimeInterval(d_integrator_time, d_integrator_time);
    auto p_regrid_projection_solver = dynamic_cast<LinearSolver*>(regrid_projection_solver.getPointer());
    if (p_regrid_projection_solver)
    {
        p_regrid_projection_solver->setInitialGuessNonzero(false);
        p_regrid_projection_solver->setNullspace(true);
    }

    // Allocate temporary data
    ComponentSelector scratch_idxs;
    scratch_idxs.setFlag(un_scr_idx);
    scratch_idxs.setFlag(us_scr_idx);
    scratch_idxs.setFlag(p_scr_idx);
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->allocatePatchData(scratch_idxs, d_integrator_time);
    }

    // Setup RHS vector
    // We need ghost cells to interpolate to nodes.
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_comp = { ITC(
        thn_cur_idx, "CONSERVATIVE_LINEAR_REFINE", true, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr) };
    HierarchyGhostCellInterpolation hier_bdry_fill;
    hier_bdry_fill.initializeOperatorState(ghost_comp, d_hierarchy);
    hier_bdry_fill.fillData(0.0);
    pre_div_interp(un_scr_idx, thn_cur_idx, un_cur_idx, us_cur_idx, d_hierarchy);
    d_hier_math_ops->div(div_idx, d_div_draw_var, 1.0, un_scr_idx, d_un_sc_var, nullptr, 0.0, true);
    ghost_comp = { ITC(div_idx, "NONE", false, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr) };
    hier_bdry_fill.deallocateOperatorState();
    hier_bdry_fill.initializeOperatorState(ghost_comp, d_hierarchy);
    hier_bdry_fill.fillData(0.0);

    const double div_U_mean = 1.0 / volume * d_hier_cc_data_ops->integral(div_idx, wgt_cc_idx);
    d_hier_cc_data_ops->addScalar(div_idx, div_idx, -div_U_mean);

    // Solve the projection poisson problem
    regrid_projection_solver->solveSystem(sol_vec, rhs_vec);
    plog << d_object_name
         << "::regridProjection(): regrid projection solve "
            "number of iterations = "
         << regrid_projection_solver->getNumIterations() << "\n";
    plog << d_object_name
         << "::regridProjection(): regrid projection solve "
            "residual norm        = "
         << regrid_projection_solver->getResidualNorm() << "\n";

    // Fill ghost cells for phi, compute grad phi, and set un := un - thn*grad(phi) and us := us - ths*grad(phi)
    ghost_comp = { ITC(
        p_scr_idx, "CONSERVATIVE_LINEAR_REFINE", true, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr) };
    hier_bdry_fill.deallocateOperatorState();
    hier_bdry_fill.initializeOperatorState(ghost_comp, d_hierarchy);
    hier_bdry_fill.fillData(0.0);

    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<CellData<NDIM, double>> phi_data = patch->getPatchData(p_scr_idx);
            Pointer<SideData<NDIM, double>> un_data = patch->getPatchData(un_cur_idx);
            Pointer<SideData<NDIM, double>> us_data = patch->getPatchData(us_cur_idx);
            Pointer<CellData<NDIM, double>> thn_data = patch->getPatchData(thn_cur_idx);

            Pointer<CartesianPatchGeometry<NDIM>> pgeom = patch->getPatchGeometry();
            const double* const dx = pgeom->getDx();

            for (int axis = 0; axis < NDIM; ++axis)
            {
                for (SideIterator<NDIM> si(patch->getBox(), axis); si; si++)
                {
                    const SideIndex<NDIM>& idx = si();
                    double grad_phi = ((*phi_data)(idx.toCell(1)) - (*phi_data)(idx.toCell(0))) / dx[axis];
                    double thn = 0.5 * ((*thn_data)(idx.toCell(1)) + (*thn_data)(idx.toCell(0)));
                    (*un_data)(idx) = (*un_data)(idx)-thn * grad_phi;
                    (*us_data)(idx) = (*us_data)(idx) - (1.0 + thn) * grad_phi;
                }
            }
        }
    }

    // Synchronize cf-interfaces
    auto sync_fcn = [&](const int dst_idx) -> void
    {
        for (int ln = d_hierarchy->getFinestLevelNumber(); ln > 0; --ln)
        {
            Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                Pointer<SideData<NDIM, double>> dst_data = patch->getPatchData(dst_idx);
                Pointer<OutersideData<NDIM, double>> os_data = patch->getPatchData(d_os_idx);
                os_data->copy(*dst_data);
            }
            Pointer<CoarsenAlgorithm<NDIM>> coarsen_alg = new CoarsenAlgorithm<NDIM>();
            Pointer<CartesianGridGeometry<NDIM>> grid_geom = d_hierarchy->getGridGeometry();
            Pointer<CoarsenOperator<NDIM>> os_coarsen_op =
                grid_geom->lookupCoarsenOperator(d_os_var, "CONSERVATIVE_COARSEN");
            coarsen_alg->registerCoarsen(dst_idx, d_os_idx, os_coarsen_op);
            std::vector<Pointer<CoarsenSchedule<NDIM>>> os_coarsen_scheds(finest_ln - coarsest_ln);
            for (int dst_ln = coarsest_ln; dst_ln < finest_ln; ++dst_ln)
            {
                Pointer<PatchLevel<NDIM>> src_level = d_hierarchy->getPatchLevel(dst_ln + 1);
                Pointer<PatchLevel<NDIM>> dst_level = d_hierarchy->getPatchLevel(dst_ln);
                os_coarsen_scheds[dst_ln] = coarsen_alg->createSchedule(dst_level, src_level);
            }
            os_coarsen_scheds[ln - 1]->coarsenData();
        }
    };

    sync_fcn(us_cur_idx);
    sync_fcn(un_cur_idx);

    // Deallocate scratch data
    for (int ln = coarsest_ln; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = d_hierarchy->getPatchLevel(ln);
        level->deallocatePatchData(scratch_idxs);
    }

    // Synchronize data on patch hierarchy
    synchronizeHierarchyData(CURRENT_DATA);
    return;
}

double
INSVCTwoFluidStaggeredHierarchyIntegrator::getStableTimestep(Pointer<Patch<NDIM>> /*patch*/) const
{
    return d_dt_init;
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::setupPlotDataSpecialized()
{
    // Interpolate velocities to node centered.
    auto var_db = VariableDatabase<NDIM>::getDatabase();
    const int un_draw_idx = var_db->mapVariableAndContextToIndex(d_un_draw_var, getCurrentContext());
    const int us_draw_idx = var_db->mapVariableAndContextToIndex(d_us_draw_var, getCurrentContext());
    const int div_draw_idx = var_db->mapVariableAndContextToIndex(d_div_draw_var, getCurrentContext());
    const int un_idx = var_db->mapVariableAndContextToIndex(d_un_sc_var, getCurrentContext());
    const int us_idx = var_db->mapVariableAndContextToIndex(d_us_sc_var, getCurrentContext());
    const int un_scr_idx = var_db->mapVariableAndContextToIndex(d_un_sc_var, getScratchContext());
    const int us_scr_idx = var_db->mapVariableAndContextToIndex(d_us_sc_var, getScratchContext());
    const int thn_cur_idx = var_db->mapVariableAndContextToIndex(d_thn_cc_var, getCurrentContext());

    const int un_rhs_idx = var_db->mapVariableAndContextToIndex(d_un_rhs_var, getCurrentContext());
    const int us_rhs_idx = var_db->mapVariableAndContextToIndex(d_us_rhs_var, getCurrentContext());
    const int un_rhs_draw_idx = var_db->mapVariableAndContextToIndex(d_un_rhs_draw_var, getCurrentContext());
    const int us_rhs_draw_idx = var_db->mapVariableAndContextToIndex(d_us_rhs_draw_var, getCurrentContext());

    static const bool synch_cf_interface = true;
    const int coarsest_ln = 0;
    const int finest_ln = d_hierarchy->getFinestLevelNumber();

    // Allocate scratch data if necessary
    allocatePatchData(un_scr_idx, 0.0, coarsest_ln, finest_ln);
    allocatePatchData(us_scr_idx, 0.0, coarsest_ln, finest_ln);

    // Set thn if necessary
    if (d_thn_fcn)
        d_thn_fcn->setDataOnPatchHierarchy(
            thn_cur_idx, d_thn_cc_var, d_hierarchy, d_integrator_time, false, coarsest_ln, finest_ln);
    if (d_thn_integrator)
    {
        // Copy thn if necessary.
        const int thn_adv_cur_idx =
            var_db->mapVariableAndContextToIndex(d_thn_cc_var, d_thn_integrator->getCurrentContext());
        d_hier_cc_data_ops->copyData(thn_cur_idx, thn_adv_cur_idx);
    }

    // We need ghost cells to interpolate to nodes.
    using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<ITC> ghost_comp = {
        ITC(un_scr_idx, un_idx, "CONSERVATIVE_LINEAR_REFINE", true, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr),
        ITC(us_scr_idx, us_idx, "CONSERVATIVE_LINEAR_REFINE", true, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr),
        ITC(thn_cur_idx, "CONSERVATIVE_LINEAR_REFINE", true, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr)
    };
    HierarchyGhostCellInterpolation hier_bdry_fill;
    hier_bdry_fill.initializeOperatorState(ghost_comp, d_hierarchy);
    hier_bdry_fill.fillData(0.0);

    // interpolate
    d_hier_math_ops->interp(un_draw_idx,
                            d_un_draw_var,
                            synch_cf_interface,
                            un_scr_idx,
                            d_un_sc_var,
                            nullptr,
                            d_integrator_time,
                            synch_cf_interface);
    d_hier_math_ops->interp(us_draw_idx,
                            d_us_draw_var,
                            synch_cf_interface,
                            us_scr_idx,
                            d_us_sc_var,
                            nullptr,
                            d_integrator_time,
                            synch_cf_interface);

    // Compute divergence of velocity fields.
    pre_div_interp(un_scr_idx, thn_cur_idx, un_scr_idx, us_scr_idx, d_hierarchy);
    d_hier_math_ops->div(div_draw_idx, d_div_draw_var, 1.0, un_scr_idx, d_un_sc_var, nullptr, 0.0, true);
    ghost_comp = { ITC(div_draw_idx, "NONE", false, "CONSERVATIVE_COARSEN", "LINEAR", false, nullptr) };
    hier_bdry_fill.deallocateOperatorState();
    hier_bdry_fill.initializeOperatorState(ghost_comp, d_hierarchy);
    hier_bdry_fill.fillData(0.0);

    {
        using ITC = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
        std::vector<ITC> ghost_comp = { ITC(un_scr_idx,
                                            un_rhs_idx,
                                            "CONSERVATIVE_LINEAR_REFINE",
                                            true,
                                            "CONSERVATIVE_COARSEN",
                                            "LINEAR",
                                            false,
                                            nullptr),
                                        ITC(us_scr_idx,
                                            us_rhs_idx,
                                            "CONSERVATIVE_LINEAR_REFINE",
                                            true,
                                            "CONSERVATIVE_COARSEN",
                                            "LINEAR",
                                            false,
                                            nullptr) };
        HierarchyGhostCellInterpolation hier_bdry_fill;
        hier_bdry_fill.initializeOperatorState(ghost_comp, d_hierarchy);
        hier_bdry_fill.fillData(0.0);
    }

    // interpolate
    d_hier_math_ops->interp(un_rhs_draw_idx,
                            d_un_rhs_draw_var,
                            synch_cf_interface,
                            un_scr_idx,
                            d_un_sc_var,
                            nullptr,
                            d_integrator_time,
                            synch_cf_interface);
    d_hier_math_ops->interp(us_rhs_draw_idx,
                            d_us_rhs_draw_var,
                            synch_cf_interface,
                            us_scr_idx,
                            d_us_sc_var,
                            nullptr,
                            d_integrator_time,
                            synch_cf_interface);

    // Deallocate scratch data
    deallocatePatchData(un_scr_idx, coarsest_ln, finest_ln);
    deallocatePatchData(us_scr_idx, coarsest_ln, finest_ln);
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::regridHierarchyBeginSpecialized()
{
    // Print out velocity data.
    pout << "Pre regrid velocities\n";
    setupPlotData();
    d_regrid_writer->writePlotData(d_hierarchy, d_regrid_write_int++, d_integrator_time);
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::regridHierarchyEndSpecialized()
{
    // Print out velocity data.
    pout << "Post regrid velocities\n";
    setupPlotData();
    d_regrid_writer->writePlotData(d_hierarchy, d_regrid_write_int++, d_integrator_time);
    pout << "Finished post regrid\n";
}

void
INSVCTwoFluidStaggeredHierarchyIntegrator::initializeCompositeHierarchyDataSpecialized(const double /*init_data_time*/,
                                                                                       const bool initial_time)
{
    // Project the interpolated velocity if needed.
    if (!initial_time)
    {
        plog << d_object_name << "::initializeCompositeHierarchyData():\n"
             << "  projecting the interpolated velocity field\n";
        regridProjection();
        setupPlotData();
        d_regrid_writer->writePlotData(d_hierarchy, d_regrid_write_int++, d_integrator_time);
    }
    return;
} // initializeCompositeHierarchyDataSpecialized

//////////////////////////////////////////////////////////////////////////////

} // namespace IBAMR

//////////////////////////////////////////////////////////////////////////////
