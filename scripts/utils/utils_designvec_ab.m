    % Helper: build design vector for a given area/state
    function x = utils_designvec_ab(ar_idx, st_idx, coef_names, area_list, state_list, ref_area, ref_state)
        x = zeros(length(coef_names),1);
        x(strcmp(coef_names,'(Intercept)')) = 1;
        if ~strcmp(area_list{ar_idx}, ref_area)
            x(strcmp(coef_names, ['Area_' area_list{ar_idx}])) = 1;
        end
        if ~strcmp(state_list{st_idx}, ref_state)
            x(strcmp(coef_names, ['State_' state_list{st_idx}])) = 1;
        end
        if ~strcmp(area_list{ar_idx}, ref_area) && ~strcmp(state_list{st_idx}, ref_state)
            x(strcmp(coef_names, ['Area_' area_list{ar_idx} ':State_' state_list{st_idx}])) = 1;
        end
    end
