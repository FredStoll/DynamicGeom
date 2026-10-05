
function fig(screen)
     % make a figure on right screen
     if screen==1
       figure("Position",[1057         482        1095         740]);
     elseif screen==2
        % figure;
         figure("Position",[3120         441         829         552]);
     % elseif screen==3
     %     figure("Position",[-1000 650 750 510]);
     end
     set(gcf,"color","white")

end
