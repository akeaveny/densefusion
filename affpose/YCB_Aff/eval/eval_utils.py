import numpy as np

#######################################
#######################################

from affpose.YCB_Aff import cfg as config
from affpose.YCB_Aff.dataset import ycb_aff_dataset_utils

#######################################
# Error Metrics.
#######################################

def get_error_metrics(gt_obj_t, gt_obj_r, pred_obj_t, pred_obj_r,
                      refinement_idx, choose, pred_c,
                      verbose=True):

    # translation
    t_error = np.linalg.norm(pred_obj_t - gt_obj_t)

    # rotation.
    error_cos = 0.5 * (np.trace(pred_obj_r @ np.linalg.inv(gt_obj_r)) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos))
    error = np.arccos(error_cos)
    R_error = 180.0 * error / np.pi

    # # ADD
    # pred = np.dot(dataloader.cld[obj_id], pred_obj_r)
    # pred = np.add(pred, pred_obj_t)
    # target = np.dot(dataloader.cld[obj_id], gt_obj_r)
    # target = np.add(target, gt_obj_t)
    # ADD = np.mean(np.linalg.norm(pred - target, axis=1))
    #
    # # ADD-S
    # tree = KDTree(pred)
    # dist, ind = tree.query(target)
    # ADD_S = np.mean(dist)

    if verbose:
        print("\tRefinement: {}, "
                "choose: {}, "
                "pred c: {:.3f}, "
                " t: {:.2f} [cm], "
                "R: {:.2f}"
              .format(refinement_idx,
                       choose,
                       pred_c,
                       t_error * 100,
                       R_error,
                       # ADD * 100,
                       # ADD_S * 100
                       ))

#######################################
# Stats.
#######################################

def get_obj_stats(pred_class_ids, objs_choose, objs_pred_c):

    # flatten arrays.
    pred_class_ids = pred_class_ids.reshape(-1)
    objs_choose = objs_choose.reshape(-1)
    objs_pred_c = objs_pred_c.reshape(-1)

    # find non zero idxs.
    non_zero_idx = np.nonzero(pred_class_ids)
    pred_class_ids = pred_class_ids[non_zero_idx]
    objs_choose = objs_choose[non_zero_idx]
    objs_pred_c = objs_pred_c[non_zero_idx]

    for obj_id in range(1, config.NUM_OBJECTS + 1):
        obj_name = "{:<25}".format(ycb_aff_dataset_utils.map_obj_id_to_name(obj_id))
        # get rows for current obj.
        idxs = np.argwhere(pred_class_ids == obj_id).reshape(-1)
        choose = np.sort(objs_choose[idxs])
        pred_c = np.sort(objs_pred_c[idxs])

        # Add correction to occlusion.
        # if np.less(min(occlusion), 0):
        #     occlusion += min(occlusion)

        # print stats.
        print('Object: {}'
              'Num Pred: {},'
              '\t\tChoose: Min: {:.0f}, '
              'Mean: {:.0f},'
              '\t\tPred C: Min: {:.3f}, '
              'Mean: {:.3f},'
              .format(obj_name,
                        len(idxs),
                        np.nanmin(choose),
                        np.nanmean(choose),
                        np.nanmin(pred_c),
                        np.nanmean(pred_c),
              ))