
%% build sdf and viz
clear all; close all;
fname = 'x_trajs_render.npy';
% fname = 'x_trajs_final.npy';
A = readNPY(fname); nf = size(A,1); np = size(A,2);
BBLL = min(reshape(A,[],3));
BBTR = max(reshape(A,[],3));
BBCC = (BBLL+BBTR)/2;
dd = (BBTR-BBCC)*1.1;
fnewBBTR = BBCC + dd;
fnewBBLL = BBCC - dd;

close all; figure; 
hold all; axis equal; rotate3d on;  view(3); 
xlim([fnewBBLL(1) fnewBBTR(1)]);
ylim([fnewBBLL(2) fnewBBTR(2)]);
zlim([fnewBBLL(3) fnewBBTR(3)]);
campos([2.5316    -0.050836       12.401]) % face side view
campos([3.26       3.3893       11.751]) % face side view
% campos([12.056     -0.60772       3.8065]) % thumb side view
camup([0 1 0])
camtarget([0 0 0])
for i=1:nf
    
    pc = reshape(A(i,:,:),[],3);
    BBLL = min(pc);
    BBTR = max(pc);
    BBCC = (BBLL+BBTR)/2;
    dd = (BBTR-BBCC)*1.8;
    newBBTR = BBCC + dd;
    newBBLL = BBCC - dd;

    N = 100000;
    lx = newBBTR(1)-newBBLL(1);
    ly = newBBTR(2)-newBBLL(2);
    lz = newBBTR(3)-newBBLL(3);
    nx = ceil((N*lx^2/(ly*lz))^(1/3));
    ny = ceil((N*ly^2/(lx*lz))^(1/3));
    nz = ceil((N*lz^2/(ly*lx))^(1/3));
    [x,y,z] = meshgrid(linspace(newBBLL(1),newBBTR(1),nx),...
                    linspace(newBBLL(2),newBBTR(2),ny),...
                    linspace(newBBLL(3),newBBTR(3),nz));
    gridxyz = [x(:) y(:) z(:)];
    [aa, sdf] = knnsearch(pc, gridxyz, 'K', 1);
    gridsdf = reshape(sdf,ny,nx,nz);
%     close all; figure; %isosurface(x,y,z,gridsdf,0); 
%     hold all; axis equal; rotate3d on; 
%     prcs = [30:5:30]/5*2;
    prcs = [40:5:50]/5*2;
    prcs = [.05:.01:.06];
    prcs = [.04:.02:.06];
    for prci = 1:numel(prcs)
        prc = prcs(prci)
%         [FF,VV] = isosurface(x,y,z,gridsdf,prctile(gridsdf(:),prc));
        [FF,VV] = isosurface(x,y,z,gridsdf,prcs(prci));
        fa = .3; if prci == 1; fa = .6; end
    %     fa = .2
        [U,Uall] = laplacian_smooth(double(VV),FF,'cotan',[],.1,'implicit',double(VV),10);
        col = [.2 .2 1]
        ptc{prci} = patch('vertices',Uall(:,:,end),'faces',FF,'facealpha',fa,'edgecolor','none','facecolor',col)
        ptc{prci}.FaceVertexCData = repmat(col,size(VV,1),1);
        
    end    
    lighting gouraud; 
    shading flat
    
    if i==1; camlight; end;
    drawnow; pause(.1)
    if i ~=nf; try for prci=1:numel(ptc); delete(ptc{prci}); end; catch; end; end;
end





