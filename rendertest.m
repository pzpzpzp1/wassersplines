%* misc matlab code snippets for various purposes *%

%% contrast test
figure; 
x = linspace(0,1,100)
% y = .95*(tanh(4*(x-.6))+1)/2;    plot(x,y); axis equal; xline(0); xline(1); ylim([0,1]); 
y = (tanh(8*(x-.5))+1)/2;    plot(x,y); axis equal; xline(0); xline(1); ylim([0,1]); 

%% RFF sigma
sigma = 2
Tperiod = 3
dim=2
tdiv=4
K0 = randn(100,1)*2*pi*sigma/sqrt(dim)/2/tdiv
Tcycles = round(K0 * Tperiod / (2*pi))
figure; hist(Tcycles)

%% try 3d volume render via matlab
clear all;
N=5000;
X = rand(N,3);
[V,F] = icosphere(2);
FF = repmat(F,1,1,N) + (reshape(1:N,1,1,N)-1)*size(V,1);
VV = .05*repmat(V,1,1,N) + reshape(X',1,3,N);
Vp = reshape(permute(VV,[1 3 2]),[],3);
Fp = reshape(permute(FF,[1 3 2]),[],3);
close all; figure; hold all; axis equal; axis off; set(gcf,'color','w'); rotate3d on;
ptc = patch('faces',Fp,'vertices',Vp,'facecolor','g','facealpha',.01,'edgecolor','none')
ptc.FaceAlpha = .0275;
ptc.FaceAlpha = .1;
ptc.FaceLighting = 'gouraud';
ptc.AmbientStrength = 0.5;
ptc.DiffuseStrength = 0.99;
ptc.SpecularStrength = 1;
ptc.SpecularExponent = 20;
% ptc.BackFaceLighting = 'unlit';
view(0,75)
ptc.FaceVertexCData = repmat([0 .447 .741],size(Vp,1),1)
% shading interp
lightangle(-45,30)



s = scatter3(X(:,1),X(:,2),X(:,3),10,'filled')
% s.MarkerFaceAlpha = 'flat'
s.MarkerFaceAlpha = .1
s.AlphaData = repmat(100,N,1) %+ .0001*randn(N,1);
s.SizeData=200
s.CData = [0 .447 .741]
% s.AlphaDataMapping = 'direct'
% s.AlphaDataMapping = 'scaled'
s.AlphaDataMapping = 'none'

%% signed curl test
close all; clear all;
v = @(x,y) [-y x]*3*pi/2;
N = 100;
dt = .001;
xx = linspace(-1,1,10);
[X,Y] = meshgrid(xx,xx);
xx = X(:); yy = Y(:);
vxxyy = v(xx,yy);
figure; hold all; axis equal;
quiver(xx,yy,vxxyy(:,1),vxxyy(:,2))

tN = 100; dt = 1/tN;
s = [0,1];
for i=1:tN
    vs = v(s(1),s(2))
    s = s + vs*dt
    scatter(s(1),s(2),'r'); drawnow;
end

%% optimal transport between oriented rectangles test
close all; clear all;
figure; axis equal; hold all; 
Area = 1; a = sqrt(Area); N = 20;
maxLfacter = 2;
ls = linspace(a,maxLfacter*a,N); % lengths
c1 = [1,0,0];c2 = [0,0,1];ctv = linspace(0,1,N)
for lind=1:N
    ct = (1-ctv(lind))*c1+c2*ctv(lind)
    l = ls(lind);
    w=Area/l;

    t = linspace(0,1,51);
    lt = l*(1-t)+t*a;
    wt = w*(1-t)+t*a;
    at = lt.*wt;
    plot(t,at,'color',ct)
end
legend(split(num2str(ls)))

%% plot areas obtained from neural traj with and w/o area preservation
a1=load('results/areas1.txt')
a2=load('results/areas2.txt')
ts = linspace(0,1,numel(a1));
f1 = figure; hold all; set(gcf,'color','w')
f1.Position= [1620.2        395.4       1034.4        327.2];
perc_increase_1 = 100*(a1/a1(1)-1);
perc_increase_2 = 100*(a2/a2(1)-1);
plot(ts,perc_increase_1,'linewidth',5)
plot(ts,perc_increase_2,'linewidth',5)
xlabel('Time(s)')
ylabel('% Area Increase')
yline(max(perc_increase_1))
yline(max(perc_increase_2))

cs = [.3 .5 1];
ct = [.2 1 .2];
ti = linspace(0,1,(numel(perc_increase_1)-1))'
colors = (1-ti).*cs + ct.*ti;
for i=1:(numel(perc_increase_1)-1)
    plot(ts(i:i+1),perc_increase_1(i:i+1),'color',.8*colors(i,:),'linewidth',5)
    plot(ts(i:i+1),perc_increase_2(i:i+1),'color',.8*colors(i,:),'linewidth',5)
end
ylim([0 45])
xticks([0,.2,.4,.6,.8,1])
yticks([0 6.3 20 40 42.7])

exportgraphics(f1, 'figures/areas.pdf');

%% sigmac to sigma
5/sqrt(2)*pi




