
%% put every mesh in a -1,1 box.
files = dir('./*.obj')
for i=1:numel(files)
    filename = files(i).name;
    [V,T] = readOBJ(filename);
    
    BB = [min(V); max(V)];
    C = (BB(1,:)+BB(2,:))/2;
    V=V-C;
    V = V / max(max(V));
    writeOBJ([filename], V, T)
end


