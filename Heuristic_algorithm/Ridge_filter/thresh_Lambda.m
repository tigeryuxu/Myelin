function [level] = thresh_Lambda(im)

%calculate bins centers
color_range = double(limits(im));

%   %try infering discrete resolution first (intensities often quantized)
%   di = min(diff(sort(unique(im(:)))));
%   num_colors = round(diff(color_range)/di)+1;
%   if num_colors>max_colors %too many levels
%     num_colors = max_colors;                %practical limit
%     di = diff(color_range)/(num_colors-1);
%   end


lo = double(color_range(1));
hi = double(color_range(2));
norm_im = (double(im)-lo)/(hi-lo);
norm_level = graythresh(norm_im); %GRAYTHRESH assumes DOUBLE range [0,1]
my_level = norm_level*(hi-lo)+lo;


level = my_level;


%----------------------------------------------------------------------
    function [x,y] = limits(a) %subfunction
        % LIMITS returns min & max values of matrix; else scalar value.
        %
        %   [lo,hi]=LIMITS(a) returns LOw and HIgh values respectively.
        %
        %   lim=LIMITS(a) returns 1x2 result, where lim = [lo hi] values
        
        if nargin~=1 | nargout>2 %bogus syntax
            error('usage: [lo,hi]=limits(a)')
        end
        
        siz=size(a);
        
        if prod(siz)==1 %scalar
            result=a;                         % value
        else %matrix
            result=[min(a(:)) max(a(:))];     % limits
        end
        
        if nargout==1 %composite result
            x=result;                         % 1x2 vector
        elseif nargout==2 %separate results
            x=result(1);                      % two scalars
            y=result(2);
        else %no result
            ans=result                        % display answer
        end
        
    end %limits (subfunction)

end