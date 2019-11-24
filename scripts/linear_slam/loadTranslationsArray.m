function t = loadTranslationsArray(file)

T = dlmread(file)';
t = [0; 0; 0; T(:)];

end
