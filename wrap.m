function varargout = wrap(functionName, varargin)
    disp(functionName);
    fun = str2func(functionName);
    returns = cell(nargout(fun), 1);
    if nargin > 1
        [returns{:}] = fun(cell2mat(varargin).data);
    else
        [returns{:}] = fun();
    end
    varargout = num2cell(arrayfun(@(r) Wrapper(r{1}), returns));
end
