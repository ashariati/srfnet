function t = loadTranslationsArray(file)

T = dlmread(file)';
t = T(:);

end
