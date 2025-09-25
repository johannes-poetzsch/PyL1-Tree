classdef Wrapper < handle
    properties
        data
    end
    methods
        function obj = Wrapper(d)
            obj.data = d;
        end
    end
end
