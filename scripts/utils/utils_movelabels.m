function utils_movelabels(coords, labels, colorareas, ax, min_dist, text_radius)

    if nargin < 4 || isempty(ax)
        ax = gca;
    end
    if nargin < 5 || isempty(min_dist)
        min_dist = 0.07; % default minimum distance from dot center to text bounding box
    end
    if nargin < 6 || isempty(text_radius)
        text_radius = 0.06; % default initial radius from dot
    end

    N = size(coords,1);
    placed_texts = zeros(N,2); % store placed text positions
    text_boxes = zeros(N,4);   % [xmin xmax ymin ymax] for each text

    % Axis units and size
    xlims = xlim(ax);
    ylims = ylim(ax);
    ax_width = diff(xlims);
    ax_height = diff(ylims);
    char_width = 0.0125 * ax_width;
    char_height = 0.03 * ax_height;

    for ar = 1:N
        fc = coords(ar,1);
        afc = coords(ar,2);
        label = labels{ar};
        nchar = length(label);
        tw = char_width * nchar;
        th = char_height;

        % Try to find a non-overlapping position
        found = false;
        for r = text_radius:0.005:text_radius+0.03
            for angle = linspace(0, 2*pi, 18)
                dx = r * cos(angle);
                dy = r * sin(angle);
                tx = fc + dx;
                ty = afc + dy;

                xmin = tx - tw/2;
                xmax = tx + tw/2;
                ymin = ty - th/2;
                ymax = ty + th/2;

                % Check dot overlap
                overlap_dot = false;
                for j = 1:N
                    if j == ar, continue; end
                    if coords(j,1) >= xmin && coords(j,1) <= xmax && ...
                       coords(j,2) >= ymin && coords(j,2) <= ymax
                        overlap_dot = true; break;
                    end
                    d = sqrt((tx-coords(j,1))^2 + (ty-coords(j,2))^2);
                    if d < min_dist + max(tw,th)/2
                        overlap_dot = true; break;
                    end
                end

                % Check previous texts overlap
                overlap_text = false;
                for k = 1:ar-1
                    box2 = text_boxes(k,:);
                    if ~(xmax < box2(1) || xmin > box2(2) || ...
                         ymax < box2(3) || ymin > box2(4))
                        overlap_text = true; break;
                    end
                end

                if ~overlap_dot && ~overlap_text
                    found = true;
                    break;
                end
            end
            if found, break; end
        end

        placed_texts(ar,:) = [tx, ty];
        text_boxes(ar,:) = [xmin xmax ymin ymax];
        text(tx, ty, label, 'Color', colorareas(ar,:)/255, ...
             'FontSize', 10, 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle');
    end
end