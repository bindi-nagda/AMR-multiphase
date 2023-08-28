#include <ibamr/app_namespaces.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/CartExtrapPhysBdryOp.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>

#include "BoxArray.h"
#include "CartesianPatchGeometry.h"
#include "CoarseFineBoundary.h"
#include "PatchGeometry.h"

#include <petscsys.h>

#include <SAMRAI_config.h>

/*******************************************************************************
 * For each run, the input filename must be given on the command line.  In all *
 * cases, the command line is:                                                 *
 *                                                                             *
 *    executable <input file name> [PETSc options]                             *
 *                                                                             *
 *******************************************************************************/
int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object as well.
    IBTKInit ibtk_init(argc, argv);

    // Parse command line options, set some standard options from the input
    // file, and enable file logging.
    Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "INS.log");
    Pointer<Database> input_db = app_initializer->getInputDatabase();

    // Retrieve "Main" section of the input database.
    Pointer<Database> main_db = app_initializer->getComponentDatabase("Main");

    int coarse_iter_num, fine_iter_num;
    if (main_db->keyExists("coarse_iter_num"))
        coarse_iter_num = main_db->getInteger("coarse_iter_num");
    else
        TBOX_ERROR("Cound not find coarse_iter_num in input file\n");
    if (main_db->keyExists("fine_iter_num"))
        fine_iter_num = main_db->getInteger("fine_iter_num");
    else
        TBOX_ERROR("Cound not find fine_iter_num in input file\n");

    string coarse_hier_dump_dirname;
    if (main_db->keyExists("coarse_hier_dump_dirname"))
        coarse_hier_dump_dirname = main_db->getString("coarse_hier_dump_dirname");
    else
        TBOX_ERROR("key `coarse_hier_dump_dirname' not specified in input file");

    string fine_hier_dump_dirname;
    if (main_db->keyExists("fine_hier_dump_dirname"))
        fine_hier_dump_dirname = main_db->getString("fine_hier_dump_dirname");
    else
        TBOX_ERROR("key `fine_hier_dump_dirname' not specified in input file");

    // Create major algorithm and data objects that comprise application.
    Pointer<CartesianGridGeometry<NDIM>> grid_geom = new CartesianGridGeometry<NDIM>(
        "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));

    // Initialize variables.
    VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();

    Pointer<VariableContext> current_ctx = var_db->getContext("FluidSolver::CURRENT");
    Pointer<VariableContext> scratch_ctx = var_db->getContext("FluidSolver::SCRATCH");
    Pointer<VariableContext> thn_ctx = var_db->getContext("AdvDiffIntegrator::CURRENT");

    Pointer<SideVariable<NDIM, double>> us_var = new SideVariable<NDIM, double>("FluidSolver::us_sc");
    const int us_idx = var_db->registerVariableAndContext(us_var, current_ctx);
    const int us_interp_idx = var_db->registerClonedPatchDataIndex(us_var, us_idx);
    const int us_scratch_idx = var_db->registerVariableAndContext(us_var, scratch_ctx, 2);

    Pointer<SideVariable<NDIM, double>> un_var = new SideVariable<NDIM, double>("FluidSolver::un_sc");
    const int un_idx = var_db->registerVariableAndContext(un_var, current_ctx);
    const int un_interp_idx = var_db->registerClonedPatchDataIndex(un_var, un_idx);
    const int un_scratch_idx = var_db->registerVariableAndContext(un_var, scratch_ctx, 2);

    Pointer<CellVariable<NDIM, double>> p_var = new CellVariable<NDIM, double>("FluidSolver::P");
    const int p_idx = var_db->registerVariableAndContext(p_var, current_ctx);
    const int p_interp_idx = var_db->registerClonedPatchDataIndex(p_var, p_idx);
    const int p_scratch_idx = var_db->registerVariableAndContext(p_var, scratch_ctx, 2);

    Pointer<CellVariable<NDIM, double>> thn_var = new CellVariable<NDIM, double>("FluidSolver::thn_cc");
    const int thn_idx = var_db->registerVariableAndContext(thn_var, thn_ctx);
    const int thn_interp_idx = var_db->registerClonedPatchDataIndex(thn_var, thn_idx);
    const int thn_scratch_idx = var_db->registerVariableAndContext(thn_var, scratch_ctx, 2);

    Pointer<CellVariable<NDIM, double>> us_draw_var = new CellVariable<NDIM, double>("us", NDIM);
    Pointer<CellVariable<NDIM, double>> un_draw_var = new CellVariable<NDIM, double>("un", NDIM);
    const int us_draw_idx = var_db->registerVariableAndContext(us_draw_var, current_ctx);
    const int un_draw_idx = var_db->registerVariableAndContext(un_draw_var, current_ctx);
    // Set up visualization plot file writer.
    Pointer<VisItDataWriter<NDIM>> visit_data_writer =
        new VisItDataWriter<NDIM>("VisIt Writer", main_db->getString("viz_dump_dirname"), 1);
    visit_data_writer->registerPlotQuantity("P", "SCALAR", p_idx);
    visit_data_writer->registerPlotQuantity("P interp", "SCALAR", p_interp_idx);
    visit_data_writer->registerPlotQuantity("Un", "VECTOR", un_draw_idx);
    visit_data_writer->registerPlotQuantity("Us", "VECTOR", us_draw_idx);
    visit_data_writer->registerPlotQuantity("thn", "SCALAR", thn_idx);
    visit_data_writer->registerPlotQuantity("thn interp", "SCALAR", thn_interp_idx);

    // Time step loop.
    double loop_time = 0.0;
    int coarse_iteration_num = coarse_iter_num;
    int fine_iteration_num = fine_iter_num;
    for (bool files_exist = true; files_exist;
         coarse_iteration_num += coarse_iter_num, fine_iteration_num += fine_iter_num)
    {
        char temp_buf[128];

        sprintf(temp_buf, "%05d.samrai.%05d", coarse_iteration_num, IBTK_MPI::getRank());
        string coarse_file_name = coarse_hier_dump_dirname + "/" + "hier_data_four_roll_mill.";
        coarse_file_name += temp_buf;

        sprintf(temp_buf, "%05d.samrai.%05d", fine_iteration_num, IBTK_MPI::getRank());
        string fine_file_name = fine_hier_dump_dirname + "/" + "hier_data_four_roll_mill.";
        fine_file_name += temp_buf;

        for (int rank = 0; rank < IBTK_MPI::getNodes(); ++rank)
        {
            if (rank == IBTK_MPI::getRank())
            {
                fstream coarse_fin, fine_fin;
                coarse_fin.open(coarse_file_name.c_str(), ios::in);
                fine_fin.open(fine_file_name.c_str(), ios::in);
                if (!coarse_fin.is_open() || !fine_fin.is_open())
                {
                    std::cout << "couldnt find file on rank " << rank << "\n";
                    std::cout << "File name was: " << coarse_file_name << "\n and " << fine_file_name << "\n";
                    files_exist = false;
                }
                else
                {
                    std::cout << "found file on rank " << rank << "\n";
                }
                coarse_fin.close();
                fine_fin.close();
            }
            IBTK_MPI::barrier();
        }

        if (!files_exist) TBOX_ERROR("Files did not exist!\n");

        pout << std::endl;
        pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        pout << "processing data" << std::endl;
        pout << "     coarse iteration number = " << coarse_iteration_num << std::endl;
        pout << "     fine iteration number = " << fine_iteration_num << std::endl;
        pout << "     coarse file name = " << coarse_file_name << std::endl;
        pout << "     fine file name = " << fine_file_name << std::endl;

        // Read in data to post-process.
        ComponentSelector hier_data;
        hier_data.setFlag(un_idx);
        hier_data.setFlag(us_idx);
        hier_data.setFlag(p_idx);
        hier_data.setFlag(thn_idx);

        Pointer<HDFDatabase> coarse_hier_db = new HDFDatabase("coarse_hier_db");
        coarse_hier_db->open(coarse_file_name);

        Pointer<PatchHierarchy<NDIM>> coarse_patch_hierarchy =
            new PatchHierarchy<NDIM>("CoarsePatchHierarchy", grid_geom, false);
        coarse_patch_hierarchy->getFromDatabase(coarse_hier_db->getDatabase("PatchHierarchy"), hier_data);

        const double coarse_loop_time = coarse_hier_db->getDouble("loop_time");

        coarse_hier_db->close();

        Pointer<HDFDatabase> fine_hier_db = new HDFDatabase("fine_hier_db");
        fine_hier_db->open(fine_file_name);

        Pointer<PatchHierarchy<NDIM>> fine_patch_hierarchy = new PatchHierarchy<NDIM>(
            "FinePatchHierarchy", grid_geom->makeRefinedGridGeometry("FineGridGeometry", 2, false), false);
        fine_patch_hierarchy->getFromDatabase(fine_hier_db->getDatabase("PatchHierarchy"), hier_data);

        const double fine_loop_time = fine_hier_db->getDouble("loop_time");

        fine_hier_db->close();

        TBOX_ASSERT(IBTK::rel_equal_eps(coarse_loop_time, fine_loop_time));
        loop_time = fine_loop_time;
        pout << "     loop time = " << loop_time << std::endl;

        Pointer<PatchHierarchy<NDIM>> coarsened_fine_patch_hierarchy =
            fine_patch_hierarchy->makeCoarsenedPatchHierarchy("CoarsenedFinePatchHierarchy", 2, false);

        // Setup hierarchy operations objects.
        HierarchyCellDataOpsReal<NDIM, double> coarse_hier_cc_data_ops(
            coarse_patch_hierarchy, 0, coarse_patch_hierarchy->getFinestLevelNumber());
        HierarchySideDataOpsReal<NDIM, double> coarse_hier_sc_data_ops(
            coarse_patch_hierarchy, 0, coarse_patch_hierarchy->getFinestLevelNumber());
        HierarchyMathOps hier_math_ops("hier_math_ops", coarse_patch_hierarchy);
        hier_math_ops.setPatchHierarchy(coarse_patch_hierarchy);
        hier_math_ops.resetLevels(0, coarse_patch_hierarchy->getFinestLevelNumber());
        const int wgt_cc_idx = hier_math_ops.getCellWeightPatchDescriptorIndex();
        const int wgt_sc_idx = hier_math_ops.getSideWeightPatchDescriptorIndex();

        // Allocate patch data.
        for (int ln = 0; ln <= coarse_patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = coarse_patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(un_interp_idx, loop_time);
            level->allocatePatchData(us_interp_idx, loop_time);
            level->allocatePatchData(p_interp_idx, loop_time);
            level->allocatePatchData(thn_interp_idx, loop_time);
            level->allocatePatchData(un_scratch_idx, loop_time);
            level->allocatePatchData(us_scratch_idx, loop_time);
            level->allocatePatchData(p_scratch_idx, loop_time);
            level->allocatePatchData(thn_scratch_idx, loop_time);
            level->allocatePatchData(un_draw_idx, loop_time);
            level->allocatePatchData(us_draw_idx, loop_time);
        }

        for (int ln = 0; ln <= fine_patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = fine_patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(un_interp_idx, loop_time);
            level->allocatePatchData(us_interp_idx, loop_time);
            level->allocatePatchData(p_interp_idx, loop_time);
            level->allocatePatchData(thn_interp_idx, loop_time);
            level->allocatePatchData(un_scratch_idx, loop_time);
            level->allocatePatchData(us_scratch_idx, loop_time);
            level->allocatePatchData(p_scratch_idx, loop_time);
            level->allocatePatchData(thn_scratch_idx, loop_time);
        }

        for (int ln = 0; ln <= coarsened_fine_patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = coarsened_fine_patch_hierarchy->getPatchLevel(ln);
            level->allocatePatchData(un_idx, loop_time);
            level->allocatePatchData(us_idx, loop_time);
            level->allocatePatchData(p_idx, loop_time);
            level->allocatePatchData(thn_idx, loop_time);
            level->allocatePatchData(un_interp_idx, loop_time);
            level->allocatePatchData(us_interp_idx, loop_time);
            level->allocatePatchData(p_interp_idx, loop_time);
            level->allocatePatchData(thn_interp_idx, loop_time);
            level->allocatePatchData(un_scratch_idx, loop_time);
            level->allocatePatchData(us_scratch_idx, loop_time);
            level->allocatePatchData(p_scratch_idx, loop_time);
            level->allocatePatchData(thn_scratch_idx, loop_time);
        }

        // Synchronize the coarse hierarchy data.
        for (int ln = coarse_patch_hierarchy->getFinestLevelNumber(); ln > 0; --ln)
        {
            Pointer<PatchLevel<NDIM>> coarser_level = coarse_patch_hierarchy->getPatchLevel(ln - 1);
            Pointer<PatchLevel<NDIM>> finer_level = coarse_patch_hierarchy->getPatchLevel(ln);

            CoarsenAlgorithm<NDIM> coarsen_alg;
            Pointer<CoarsenOperator<NDIM>> coarsen_op;

            coarsen_op = grid_geom->lookupCoarsenOperator(un_var, "CONSERVATIVE_COARSEN");
            coarsen_alg.registerCoarsen(un_idx, un_idx, coarsen_op);

            coarsen_op = grid_geom->lookupCoarsenOperator(us_var, "CONSERVATIVE_COARSEN");
            coarsen_alg.registerCoarsen(us_idx, us_idx, coarsen_op);

            coarsen_op = grid_geom->lookupCoarsenOperator(p_var, "CONSERVATIVE_COARSEN");
            coarsen_alg.registerCoarsen(p_idx, p_idx, coarsen_op);

            coarsen_op = grid_geom->lookupCoarsenOperator(thn_var, "CONSERVATIVE_COARSEN");
            coarsen_alg.registerCoarsen(thn_idx, thn_idx, coarsen_op);

            coarsen_alg.createSchedule(coarser_level, finer_level)->coarsenData();
        }

        // Synchronize the fine hierarchy data.
        for (int ln = fine_patch_hierarchy->getFinestLevelNumber(); ln > 0; --ln)
        {
            Pointer<PatchLevel<NDIM>> coarser_level = fine_patch_hierarchy->getPatchLevel(ln - 1);
            Pointer<PatchLevel<NDIM>> finer_level = fine_patch_hierarchy->getPatchLevel(ln);

            CoarsenAlgorithm<NDIM> coarsen_alg;
            Pointer<CoarsenOperator<NDIM>> coarsen_op;

            coarsen_op = grid_geom->lookupCoarsenOperator(un_var, "CONSERVATIVE_COARSEN");
            coarsen_alg.registerCoarsen(un_idx, un_idx, coarsen_op);

            coarsen_op = grid_geom->lookupCoarsenOperator(us_var, "CONSERVATIVE_COARSEN");
            coarsen_alg.registerCoarsen(us_idx, us_idx, coarsen_op);

            coarsen_op = grid_geom->lookupCoarsenOperator(p_var, "CONSERVATIVE_COARSEN");
            coarsen_alg.registerCoarsen(p_idx, p_idx, coarsen_op);

            coarsen_op = grid_geom->lookupCoarsenOperator(thn_var, "CONSERVATIVE_COARSEN");
            coarsen_alg.registerCoarsen(thn_idx, thn_idx, coarsen_op);

            coarsen_alg.createSchedule(coarser_level, finer_level)->coarsenData();
        }

        // Coarsen data from the fine hierarchy to the coarsened fine hierarchy.
        for (int ln = 0; ln <= fine_patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> dst_level = coarsened_fine_patch_hierarchy->getPatchLevel(ln);
            Pointer<PatchLevel<NDIM>> src_level = fine_patch_hierarchy->getPatchLevel(ln);

            Pointer<CoarsenOperator<NDIM>> coarsen_op;
            for (PatchLevel<NDIM>::Iterator p(dst_level); p; p++)
            {
                Pointer<Patch<NDIM>> dst_patch = dst_level->getPatch(p());
                Pointer<Patch<NDIM>> src_patch = src_level->getPatch(p());
                const Box<NDIM>& coarse_box = dst_patch->getBox();
                TBOX_ASSERT(Box<NDIM>::coarsen(src_patch->getBox(), 2) == coarse_box);

                coarsen_op = grid_geom->lookupCoarsenOperator(un_var, "CONSERVATIVE_COARSEN");
                coarsen_op->coarsen(*dst_patch, *src_patch, un_interp_idx, un_idx, coarse_box, 2);

                coarsen_op = grid_geom->lookupCoarsenOperator(us_var, "CONSERVATIVE_COARSEN");
                coarsen_op->coarsen(*dst_patch, *src_patch, us_interp_idx, us_idx, coarse_box, 2);

                coarsen_op = grid_geom->lookupCoarsenOperator(p_var, "CONSERVATIVE_COARSEN");
                coarsen_op->coarsen(*dst_patch, *src_patch, p_interp_idx, p_idx, coarse_box, 2);

                coarsen_op = grid_geom->lookupCoarsenOperator(thn_var, "CONSERVATIVE_COARSEN");
                coarsen_op->coarsen(*dst_patch, *src_patch, thn_interp_idx, thn_idx, coarse_box, 2);
            }
        }

        // Interpolate and copy data from the coarsened fine patch hierarchy to
        // the coarse patch hierarchy.
        for (int ln = 0; ln <= coarse_patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            pout << "Interpolating on level " << ln << "\n";
            Pointer<PatchLevel<NDIM>> dst_level = coarse_patch_hierarchy->getPatchLevel(ln);
            Pointer<PatchLevel<NDIM>> src_level = coarsened_fine_patch_hierarchy->getPatchLevel(ln);

            RefineAlgorithm<NDIM> refine_alg;
            Pointer<RefineOperator<NDIM>> refine_op;

            refine_op = grid_geom->lookupRefineOperator(un_var, "CONSERVATIVE_LINEAR_REFINE");
            refine_alg.registerRefine(un_interp_idx, un_interp_idx, un_scratch_idx, refine_op);

            refine_op = grid_geom->lookupRefineOperator(us_var, "CONSERVATIVE_LINEAR_REFINE");
            refine_alg.registerRefine(us_interp_idx, us_interp_idx, us_scratch_idx, refine_op);

            refine_op = grid_geom->lookupRefineOperator(p_var, "LINEAR_REFINE");
            refine_alg.registerRefine(p_interp_idx, p_interp_idx, p_scratch_idx, refine_op);

            refine_op = grid_geom->lookupRefineOperator(thn_var, "LINEAR_REFINE");
            refine_alg.registerRefine(thn_interp_idx, thn_interp_idx, thn_scratch_idx, refine_op);

            ComponentSelector data_indices;
            data_indices.setFlag(un_scratch_idx);
            data_indices.setFlag(us_scratch_idx);
            data_indices.setFlag(p_scratch_idx);
            data_indices.setFlag(thn_scratch_idx);
            CartExtrapPhysBdryOp bc_helper(data_indices, "LINEAR");

            refine_alg.createSchedule(dst_level, src_level, ln - 1, coarse_patch_hierarchy, &bc_helper)
                ->fillData(loop_time);
        }

        for (int ln = 0; ln <= coarse_patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = coarse_patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                const Box<NDIM>& box = patch->getBox();
                Pointer<CellData<NDIM, double>> un_draw_data = patch->getPatchData(un_draw_idx);
                Pointer<SideData<NDIM, double>> un_data = patch->getPatchData(un_interp_idx);
                Pointer<CellData<NDIM, double>> us_draw_data = patch->getPatchData(us_draw_idx);
                Pointer<SideData<NDIM, double>> us_data = patch->getPatchData(us_interp_idx);
                for (CellIterator<NDIM> i(box); i; i++)
                {
                    CellIndex<NDIM> idx = i();
                    SideIndex<NDIM> idx_b(idx, 1, 0);
                    SideIndex<NDIM> idx_u(idx, 1, 1);
                    SideIndex<NDIM> idx_l(idx, 0, 0);
                    SideIndex<NDIM> idx_r(idx, 0, 1);
                    (*un_draw_data)(idx, 0) = 0.5 * ((*un_data)(idx_l) + (*un_data)(idx_r));
                    (*un_draw_data)(idx, 1) = 0.5 * ((*un_data)(idx_b) + (*un_data)(idx_u));
                    (*us_draw_data)(idx, 0) = 0.5 * ((*us_data)(idx_l) + (*us_data)(idx_r));
                    (*us_draw_data)(idx, 1) = 0.5 * ((*us_data)(idx_b) + (*us_data)(idx_u));
                }
            }
        }

        // Output plot data before taking norms of differences.
        visit_data_writer->writePlotData(coarse_patch_hierarchy, coarse_iteration_num, loop_time);

        // Compute norms of differences.
        coarse_hier_sc_data_ops.subtract(un_interp_idx, un_idx, un_interp_idx);
        coarse_hier_sc_data_ops.subtract(us_interp_idx, us_idx, us_interp_idx);
        coarse_hier_cc_data_ops.subtract(p_interp_idx, p_idx, p_interp_idx);
        coarse_hier_cc_data_ops.subtract(thn_interp_idx, thn_idx, thn_interp_idx);

        for (int ln = 0; ln <= coarse_patch_hierarchy->getFinestLevelNumber(); ++ln)
        {
            Pointer<PatchLevel<NDIM>> level = coarse_patch_hierarchy->getPatchLevel(ln);
            for (PatchLevel<NDIM>::Iterator p(level); p; p++)
            {
                Pointer<Patch<NDIM>> patch = level->getPatch(p());
                const Box<NDIM>& box = patch->getBox();
                Pointer<CellData<NDIM, double>> un_draw_data = patch->getPatchData(un_draw_idx);
                Pointer<SideData<NDIM, double>> un_data = patch->getPatchData(un_interp_idx);
                Pointer<CellData<NDIM, double>> us_draw_data = patch->getPatchData(us_draw_idx);
                Pointer<SideData<NDIM, double>> us_data = patch->getPatchData(us_interp_idx);
                for (CellIterator<NDIM> i(box); i; i++)
                {
                    CellIndex<NDIM> idx = i();
                    SideIndex<NDIM> idx_b(idx, 1, 0);
                    SideIndex<NDIM> idx_u(idx, 1, 1);
                    SideIndex<NDIM> idx_l(idx, 0, 0);
                    SideIndex<NDIM> idx_r(idx, 0, 1);
                    (*un_draw_data)(idx, 0) = 0.5 * ((*un_data)(idx_l) + (*un_data)(idx_r));
                    (*un_draw_data)(idx, 1) = 0.5 * ((*un_data)(idx_b) + (*un_data)(idx_u));
                    (*us_draw_data)(idx, 0) = 0.5 * ((*us_data)(idx_l) + (*us_data)(idx_r));
                    (*us_draw_data)(idx, 1) = 0.5 * ((*us_data)(idx_b) + (*us_data)(idx_u));
                }
            }
        }

        pout << "\n"
             << "Error in " << thn_var->getName() << " at time " << loop_time << ":\n"
             << "  L1-norm:  " << coarse_hier_cc_data_ops.L1Norm(thn_interp_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << coarse_hier_cc_data_ops.L2Norm(thn_interp_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << coarse_hier_cc_data_ops.maxNorm(thn_interp_idx, wgt_cc_idx) << "\n";

        pout << "\n"
             << "Error in " << un_var->getName() << " at time " << loop_time << ":\n"
             << "  L1-norm:  " << coarse_hier_sc_data_ops.L1Norm(un_interp_idx, wgt_sc_idx) << "\n"
             << "  L2-norm:  " << coarse_hier_sc_data_ops.L2Norm(un_interp_idx, wgt_sc_idx) << "\n"
             << "  max-norm: " << coarse_hier_sc_data_ops.maxNorm(un_interp_idx, wgt_sc_idx) << "\n";

        pout << "\n"
             << "Error in " << us_var->getName() << " at time " << loop_time << ":\n"
             << "  L1-norm:  " << coarse_hier_sc_data_ops.L1Norm(us_interp_idx, wgt_sc_idx) << "\n"
             << "  L2-norm:  " << coarse_hier_sc_data_ops.L2Norm(us_interp_idx, wgt_sc_idx) << "\n"
             << "  max-norm: " << coarse_hier_sc_data_ops.maxNorm(us_interp_idx, wgt_sc_idx) << "\n";

        pout << "\n"
             << "Error in " << p_var->getName() << " at time " << loop_time << ":\n"
             << "  L1-norm:  " << coarse_hier_cc_data_ops.L1Norm(p_interp_idx, wgt_cc_idx) << "\n"
             << "  L2-norm:  " << coarse_hier_cc_data_ops.L2Norm(p_interp_idx, wgt_cc_idx) << "\n"
             << "  max-norm: " << coarse_hier_cc_data_ops.maxNorm(p_interp_idx, wgt_cc_idx) << "\n";

        // Output plot data after taking norms of differences.
        visit_data_writer->writePlotData(coarse_patch_hierarchy, coarse_iteration_num + 1, loop_time);

        pout << std::endl;
        pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        pout << std::endl;
    }
    return 0;
} // main
