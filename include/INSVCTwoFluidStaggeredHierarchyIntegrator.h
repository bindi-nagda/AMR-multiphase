// ---------------------------------------------------------------------
//
// Copyright (c) 2008 - 2022 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_IBAMR_INSVCTwoFluidStaggeredHierarchyIntegrator
#define included_IBAMR_INSVCTwoFluidStaggeredHierarchyIntegrator

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

#include "ibamr/INSHierarchyIntegrator.h"
#include "ibamr/StaggeredStokesPhysicalBoundaryHelper.h"
#include "ibamr/StaggeredStokesSolver.h"
#include "ibamr/StaggeredStokesSolverManager.h"
#include "ibamr/ibamr_enums.h"

#include "ibtk/SideDataSynchronization.h"
#include <ibtk/muParserCartGridFunction.h>

#include "CellVariable.h"
#include "HierarchyCellDataOpsReal.h"
#include "HierarchyFaceDataOpsReal.h"
#include "HierarchySideDataOpsReal.h"
#include "IntVector.h"
#include "MultiblockDataTranslator.h"
#include "SAMRAIVectorReal.h"
#include "SideVariable.h"
#include "tbox/Database.h"
#include "tbox/Pointer.h"

#include <string>
#include <vector>

// Local includes
#include "FullFACPreconditioner.h"
#include "VCTwoFluidStaggeredStokesBoxRelaxationFACOperator.h"
#include "VCTwoFluidStaggeredStokesOperator.h"

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace IBAMR
{
/*!
 * \brief Class INSVCTwoFluidStaggeredHierarchyIntegrator provides a staggered-grid solver
 * for the incompressible Navier-Stokes equations on an AMR grid hierarchy.
 */
class INSVCTwoFluidStaggeredHierarchyIntegrator : public INSHierarchyIntegrator
{
public:
    /*!
     * The constructor for class INSVCTwoFluidStaggeredHierarchyIntegrator sets some
     * default values, reads in configuration information from input and restart
     * databases, and registers the integrator object with the restart manager
     * when requested.
     */
    INSVCTwoFluidStaggeredHierarchyIntegrator(
        std::string object_name,
        SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
        SAMRAI::tbox::Pointer<SAMRAI::geom::CartesianGridGeometry<NDIM>> grid_geometry,
        bool register_for_restart = true);

    /*!
     * The destructor for class INSVCTwoFluidStaggeredHierarchyIntegrator unregisters the
     * integrator object with the restart manager when the object is so
     * registered.
     */
    ~INSVCTwoFluidStaggeredHierarchyIntegrator();

    /*!
     * Non-zero Reynolds numbers are not implemented. Returns nullptr.
     */
    SAMRAI::tbox::Pointer<ConvectiveOperator> getConvectiveOperator() override;

    /*!
     * Not in use. Returns nullptr.
     */
    SAMRAI::tbox::Pointer<IBTK::PoissonSolver> getVelocitySubdomainSolver() override;

    /*!
     * Not in use. Returns nullptr.
     */
    SAMRAI::tbox::Pointer<IBTK::PoissonSolver> getPressureSubdomainSolver() override;

    /*!
     * Get solvent velocity variable.
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> getSolventVariable() const;

    /*!
     * Get network velocity variable.
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> getNetworkVariable() const;

    /*!
     * Get pressure velocity variable.
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> getPressureVariable() const;

    inline SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> getNetworkVolumeFractionVariable() const
    {
        return d_thn_cc_var;
    }

    /*!
     * \brief Set the viscosity coefficients for the viscous stresses.
     */
    void setViscosityCoefficient(double eta_n, double eta_s);

    /*!
     * \brief Set the drag coefficients for each phase.
     */
    void setDragCoefficient(double xi, double nu_n, double nu_s);

    /*!
     * Set initial conditions for the state variables.
     *
     * NOTE: These pointers are set to nullptr after they are used.
     */
    void setInitialData(SAMRAI::tbox::Pointer<IBTK::CartGridFunction> un_fcn,
                        SAMRAI::tbox::Pointer<IBTK::CartGridFunction> us_fcn,
                        SAMRAI::tbox::Pointer<IBTK::CartGridFunction> p_fcn);

    /*!
     * Set the initial conditions for the network volume fraction.
     */
    void setInitialNetworkVolumeFraction(SAMRAI::tbox::Pointer<IBTK::CartGridFunction> thn_init_fcn);

    /*!
     * Register forcing functions. These are not scaled by the volume fraction. Any can be NULL.
     */
    void setForcingFunctions(SAMRAI::tbox::Pointer<IBTK::CartGridFunction> fn_fcn,
                             SAMRAI::tbox::Pointer<IBTK::CartGridFunction> fs_fcn,
                             SAMRAI::tbox::Pointer<IBTK::CartGridFunction> fp_fcn = nullptr);

    /*!
     * Regiser forcing functions that should be scaled by the volume fraction.
     */
    void setForcingFunctionsScaled(SAMRAI::tbox::Pointer<IBTK::CartGridFunction> fn_fcn,
                                   SAMRAI::tbox::Pointer<IBTK::CartGridFunction> fs_fcn);

    /*!
     * Register the volume fraction function. If a function is not registered, the volume fraction is advected with the
     * fluid.
     *
     * An optional argument allows this function to be used as the initial condition for the volume fraction. If this is
     * set to false, users are expected to call setInitialNetworkVolumeFraction.
     */
    void setNetworkVolumeFractionFunction(SAMRAI::tbox::Pointer<IBTK::CartGridFunction> thn_fcn,
                                          bool use_as_initial_condition = true);

    /*!
     * Sets that the network volume fraction should be advected with the specified advection diffusion integrator. Also
     * registers the integrator with the hierarchy.
     */
    void advectNetworkVolumeFraction(SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> adv_diff_integrator);

    /*!
     * Initialize the variables, basic communications algorithms, solvers, and
     * other data structures used by this time integrator object.
     *
     * This method is called automatically by initializePatchHierarchy() prior
     * to the construction of the patch hierarchy.  It is also possible for
     * users to make an explicit call to initializeHierarchyIntegrator() prior
     * to calling initializePatchHierarchy().
     */
    void
    initializeHierarchyIntegrator(SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                  SAMRAI::tbox::Pointer<SAMRAI::mesh::GriddingAlgorithm<NDIM>> gridding_alg) override;

    /*!
     * Virtual method to perform implementation-specific data initialization on
     * a new level after it is inserted into an AMR patch hierarchy by the
     * gridding algorithm.
     *
     * An empty default implementation is provided.
     */
    void initializeLevelDataSpecialized(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                                        int level_number,
                                        double init_data_time,
                                        bool can_be_refined,
                                        bool initial_time,
                                        SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchLevel<NDIM>> old_level,
                                        bool allocate_data) override;

    /*!
     * Prepare to advance the data from current_time to new_time.
     */
    void preprocessIntegrateHierarchy(double current_time, double new_time, int num_cycles = 1) override;

    /*!
     * Synchronously advance each level in the hierarchy over the given time
     * increment.
     */
    void integrateHierarchy(double current_time, double new_time, int cycle_num = 0) override;

    /*!
     * Clean up data following call(s) to integrateHierarchy().
     */
    void postprocessIntegrateHierarchy(double current_time,
                                       double new_time,
                                       bool skip_synchronize_new_state_data,
                                       int num_cycles = 1) override;

    void regridProjection() override;

    double getStableTimestep(SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch) const override;

    /*!
     * Reset cached hierarchy dependent data.
     */
    void resetHierarchyConfigurationSpecialized(SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                                                int coarsest_level,
                                                int finest_level) override;

    void applyGradientDetectorSpecialized(const SAMRAI::tbox::Pointer<SAMRAI::hier::BasePatchHierarchy<NDIM>> hierarchy,
                                          int level_num,
                                          double error_data_time,
                                          int tag_idx,
                                          bool initial_time,
                                          bool uses_richardson_extrapolation_too);

protected:
    void setupPlotDataSpecialized() override;

    void regridHierarchyBeginSpecialized() override;

    void regridHierarchyEndSpecialized() override;

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    INSVCTwoFluidStaggeredHierarchyIntegrator() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    INSVCTwoFluidStaggeredHierarchyIntegrator(const INSVCTwoFluidStaggeredHierarchyIntegrator& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    INSVCTwoFluidStaggeredHierarchyIntegrator&
    operator=(const INSVCTwoFluidStaggeredHierarchyIntegrator& that) = delete;

    SAMRAI::tbox::Pointer<IBTK::CartGridFunction> d_f_un_fcn, d_f_us_fcn, d_f_p_fcn;
    SAMRAI::tbox::Pointer<IBTK::CartGridFunction> d_f_un_thn_fcn, d_f_us_ths_fcn;

    // CartGridFunctions that set the initial values for state variables.
    SAMRAI::tbox::Pointer<IBTK::CartGridFunction> d_un_init_fcn, d_us_init_fcn, d_p_init_fcn, d_thn_init_fcn;

    /*!
     * Fluid solver variables.
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_un_sc_var;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_us_sc_var;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_un_rhs_var, d_us_rhs_var;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_p_rhs_var;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_f_un_sc_var;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_f_us_sc_var;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_f_cc_var;

    /*!
     * Gradient detector variables
     */
    bool d_use_grad_tagging = false;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_grad_thn_var;
    SAMRAI::tbox::Array<double> d_abs_grad_thresh, d_rel_grad_thresh;
    double d_max_grad_thn = std::numeric_limits<double>::quiet_NaN();

    /*!
     * Vectors
     */
    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_sol_vec;
    SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>> d_rhs_vec;
    std::vector<SAMRAI::tbox::Pointer<SAMRAI::solv::SAMRAIVectorReal<NDIM, double>>> d_nul_vecs;

    // Physical parameters
    double d_rho = std::numeric_limits<double>::quiet_NaN();
    double d_xi = std::numeric_limits<double>::quiet_NaN(), d_nu_n = std::numeric_limits<double>::quiet_NaN(),
           d_nu_s = std::numeric_limits<double>::quiet_NaN();
    double d_eta_n = std::numeric_limits<double>::quiet_NaN(), d_eta_s = std::numeric_limits<double>::quiet_NaN();

    /*!
     * Solver information
     */
    SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> d_solver_db;
    SAMRAI::tbox::Pointer<IBTK::PETScKrylovLinearSolver> d_stokes_solver;
    SAMRAI::tbox::Pointer<VCTwoFluidStaggeredStokesOperator> d_stokes_op;

    /*!
     * Preconditioner information
     */
    SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> d_precond_db;
    SAMRAI::tbox::Pointer<IBTK::FullFACPreconditioner> d_stokes_precond;
    SAMRAI::tbox::Pointer<VCTwoFluidStaggeredStokesBoxRelaxationFACOperator> d_precond_op;
    double d_w = std::numeric_limits<double>::quiet_NaN();
    bool d_use_preconditioner = true;

    /*!
     * Velocity Drawing information.
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_un_draw_var, d_us_draw_var;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_div_draw_var;

    /*!
     * Objects that can do the operations we need
     */
    SAMRAI::tbox::Pointer<SAMRAI::math::HierarchyCellDataOpsReal<NDIM, double>> d_hier_cc_data_ops;
    SAMRAI::tbox::Pointer<SAMRAI::math::HierarchySideDataOpsReal<NDIM, double>> d_hier_sc_data_ops;
    SAMRAI::tbox::Pointer<SAMRAI::math::HierarchyFaceDataOpsReal<NDIM, double>> d_hier_fc_data_ops;

    /*!
     * Network volume fraction. We either specify the function thn_fcn, or we use the d_thn_integrator.
     *
     * IMPORTANT: d_thn_integrator should be listed in d_adv_diff_hier_integrators. We ONLY use d_thn_integrator for the
     * contexts.
     */
    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_thn_cc_var;
    SAMRAI::tbox::Pointer<IBTK::CartGridFunction> d_thn_fcn;
    SAMRAI::tbox::Pointer<IBAMR::AdvDiffHierarchyIntegrator> d_thn_integrator;

    bool d_make_div_rhs_sum_to_zero = true;

    // DEBUG regridding visit writer.
    SAMRAI::tbox::Pointer<SAMRAI::pdat::NodeVariable<NDIM, double>> d_un_rhs_draw_var, d_us_rhs_draw_var;
    SAMRAI::tbox::Pointer<SAMRAI::appu::VisItDataWriter<NDIM>> d_regrid_writer;
    int d_regrid_write_int = 0;
};
} // namespace IBAMR

//////////////////////////////////////////////////////////////////////////////

#endif // #ifndef included_IBAMR_INSVCTwoFluidStaggeredHierarchyIntegrator
