function t = rotateTranslationWindow(windowOrientation, translations, conn)

numWindows = size(windowOrientation, 3) - (conn-1);

translationIndex = 0;
t = zeros(size(translations));
for i=1:numWindows
    
    Ri = windowOrientation(:, :, i);
    
    for j=1:(conn-1)
        
        k = (3*translationIndex) + 1;
        t(k:k+2) = Ri * translations(k:k+2);
        
        translationIndex = translationIndex + 1;
        
    end

end
