function allChoices = choosedialog2()

% Other == 1, 5, 2, 3, 4 
% AND April 12

% BCOR == 

pop1s ={'whole';'DAPI';'Red Channel';'Fibers';'Green Field';};
pop2s = {'Fibers';'Red Channel'; 'DAPI';'Green Field';'whole'};
pop3s = {'Red Channel';'DAPI';'Fibers';'Green Field';'whole'};
pop4s = {'DAPI';'Green Field';'Red Channel';'Fibers';'whole'};
pop5s = {'Green Field';'DAPI';'Red Channel';'Fibers';'whole'};

allChoices = [pop1s(1), pop2s(1), pop3s(1), pop4s(1), pop5s(1)];

d = dialog('Position',[300 300 250 250],'Name','Which channels are you running?');
txt = uicontrol('Parent',d,...
    'Style','text',...
    'Position',[20 200 210 40],...
    'String','Which channels are you running?');

popup1  = uicontrol('Parent',d,...)
    'Style','popup',...
    'Position',[75 180 100 25],...
    'String',pop1s,...
    'Callback',@popup_callback1);

popup2 = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[75 150 100 25],...
    'String',pop2s,...
    'Callback',@popup_callback2);

popup3 = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[75 120 100 25],...
    'String',pop3s,...
    'Callback',@popup_callback3);

popup4 = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[75 90 100 25],...
    'String',pop4s,...
    'Callback',@popup_callback4);


popup5 = uicontrol('Parent',d,...
    'Style','popup',...
    'Position',[75 60 100 25],...
    'String', pop5s,...
    'Callback',@popup_callback5);

btn = uicontrol('Parent',d,...
    'Position',[89 30 70 25],...
    'String','OK',...
    'Callback','delete(gcf)');




%     Wait for d to close before running to completion
uiwait(d);

    function popup_callback1(popup,event)
        idx = popup.Value
        popup_items = popup.String;
        choice = char(popup_items(idx,:))
        x = rand(5, 1);
        allChoices{1} = choice;
    end
    function popup_callback2(popup,event)
        idx = popup.Value
        popup_items = popup.String;
        choice = char(popup_items(idx,:))
        x = rand(5, 1);
        allChoices{2} = choice;
    end
    function popup_callback3(popup,event)
        idx = popup.Value
        popup_items = popup.String;
        choice = char(popup_items(idx,:))
        x = rand(5, 1);
        allChoices{3} = choice;
    end
    function popup_callback4(popup,event)
        idx = popup.Value
        popup_items = popup.String;
        choice = char(popup_items(idx,:))
        x = rand(5, 1);
        allChoices{4} = choice;
    end
    function popup_callback5(popup,event)
        idx = popup.Value
        popup_items = popup.String;
        choice = char(popup_items(idx,:))
        x = rand(5, 1);
        allChoices{5} = choice;
    end
end




