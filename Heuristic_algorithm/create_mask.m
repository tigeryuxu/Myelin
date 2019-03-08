
function [mask] = create_mask(x_size, y_size, width, height)

        total_length_x = 400;
        total_length_y = 400;
        
        x_left = x_size - total_length_x / 2;
        x_right = x_size + total_length_x / 2;
        
        % adaptive cropping for width (x-axis)
        if x_left <= 0
            x_right = x_right + abs(x_left) + 1;
            x_left = 1;
            
        elseif x_right > width
            x_left = x_left - (x_right - width);
            x_right = width;
        end
        
        % adaptive cropping for height (y-axis)
        y_top = y_size - total_length_y / 2;
        y_bottom = y_size + total_length_y / 2;
        if y_top <= 0
            y_bottom = y_bottom + abs(y_top) + 1;
            y_top = 1;
            
        elseif y_bottom > height
            y_top = y_top - (y_bottom - height);
            y_bottom = height;
        end
        
        % Final check to see if sizes are correct
        if (x_right - x_left) ~= 100 || (y_bottom - y_top) ~= 100
            %break;
            j = 'ERROR in crop size';
        end
        
        mask = zeros(size(core_dist));
        mask(x_left:x_right, y_top:y_bottom) = 1;
end