%% EXTRACTS all nodal distances
function [all_nodal_dist] = get_nodal_distances(caspr_image, largest_distance)
    cc = bwconncomp(caspr_image);
    centers = regionprops3(cc, 'VoxelIdxList', 'Centroid');
    all_nodal_dist = [];
    for node_idx = 1:length(centers.VoxelIdxList)
        tmp = zeros(size(caspr_image));
        center_list = centers.VoxelIdxList{node_idx};
        tmp(center_list) = 1;
        dist = bwdist(tmp);
        dist(caspr_image < 1) = 0;
        unique_vals = unique(dist);
        if length(unique_vals) > 1
            smallest_val = unique_vals(2);
            if smallest_val < largest_distance
                all_nodal_dist = [all_nodal_dist, smallest_val];
            end
        end
    end
end