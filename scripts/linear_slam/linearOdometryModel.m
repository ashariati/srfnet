function A = linearOdometryModel(numPoints, conn, stride)

connBlock = zeros(conn-1, conn);
connBlock(:, 1) = -1;
connBlock(:, 2:end) = eye(conn-1);

numBlocks = floor((numPoints - (conn-1)) / stride);

modelHeight = numBlocks * (conn-1);
modelWidth = numPoints;
A = sparse(modelHeight, modelWidth);

dh = (conn-1) - 1;
dw = conn - 1;
for b=0:(numBlocks-1)
    
    i = ((conn-1) * b) + 1;
    j = (b * (stride)) + 1;
    
    A(i:i+dh, j:j+dw) = connBlock;

end

A0 = sparse(1, modelWidth);
A0(1, 1) = 1;
A = [A0; A];

A = kron(A, eye(3));

