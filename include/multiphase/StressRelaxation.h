#ifndef included_multiphase_StressRelaxation
#define included_multiphase_StressRelaxation

#include <ibamr/config.h>

#include "ibamr/CFRelaxationOperator.h"
#include "ibamr/ibamr_enums.h"

#include "Box.h"
#include "CartesianPatchGeometry.h"
#include "CellData.h"
#include "CellIndex.h"
#include "CellVariable.h"
#include "Index.h"
#include "IntVector.h"
#include "Patch.h"
#include "PatchGeometry.h"
#include "PatchHierarchy.h"
#include "PatchLevel.h"
#include "Variable.h"
#include "tbox/Database.h"
#include "tbox/Pointer.h"
#include "tbox/Utilities.h"

#include <string>

namespace multiphase
{
/*!
 * \brief Class StressRelaxation is a concrete CFRelaxationOperator that computes the relaxation function for an
 * Oldroyd-B fluid model.
 */
class StressRelaxation : public IBAMR::CFRelaxationOperator
{
public:
    /*!
     * \brief This constructor reads in the parameters for the model from the input database.
     */
    StressRelaxation(const std::string& object_name,
                     SAMRAI::tbox::Pointer<SAMRAI::tbox::Database> input_db,
                     SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> u_var,
                     SAMRAI::tbox::Pointer<IBTK::HierarchyIntegrator> u_integrator,
                     SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> thn_var,
                     SAMRAI::tbox::Pointer<IBTK::HierarchyIntegrator> thn_integrator);

    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    StressRelaxation() = delete;

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    StressRelaxation(const StressRelaxation& from) = delete;

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     */
    StressRelaxation& operator=(const StressRelaxation& that) = delete;

    /*!
     * \brief Empty destructor.
     */
    ~StressRelaxation() = default;

    /*!
     * \name Methods to set patch data.
     */
    //\{

    void setDataOnPatchHierarchy(const int data_idx,
                                 SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> var,
                                 SAMRAI::tbox::Pointer<SAMRAI::hier::PatchHierarchy<NDIM>> hierarchy,
                                 const double data_time,
                                 const bool initial_time = false,
                                 const int coarsest_ln = IBTK::invalid_level_number,
                                 const int finest_ln = IBTK::invalid_level_number) override;

    /*!
     * \brief Evaluate the function on the patch interior.
     */
    void setDataOnPatch(const int data_idx,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM>> var,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::Patch<NDIM>> patch,
                        const double data_time,
                        const bool initial_time = false,
                        SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>> patch_level =
                            SAMRAI::tbox::Pointer<SAMRAI::hier::PatchLevel<NDIM>>(NULL)) override;

    //\}

private:
    double d_lambda = std::numeric_limits<double>::quiet_NaN();
    double d_mu = std::numeric_limits<double>::quiet_NaN();

    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_EE_var;
    int d_EE_idx;
    SAMRAI::tbox::Pointer<SAMRAI::pdat::SideVariable<NDIM, double>> d_u_var;
    SAMRAI::tbox::Pointer<IBTK::HierarchyIntegrator> d_u_integrator;

    SAMRAI::tbox::Pointer<SAMRAI::pdat::CellVariable<NDIM, double>> d_thn_var;
    SAMRAI::tbox::Pointer<IBTK::HierarchyIntegrator> d_thn_integrator;
};
} // namespace multiphase
#endif