import cv2 as cv
import glob
import os
from geomloss import SamplesLoss
import numpy as np
import matplotlib
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import matplotlib.animation
import torch
import trimesh
import ot
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import Figure, Layout, Scatter3d
from scipy.spatial.distance import squareform
from torch import nn
import pdb
import scipy.interpolate as scipyinterpolate
from meshpy.tet import MeshInfo, build


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ezshow(dat, col='green'):
    ax = plt.gca()
    datp = dat.detach().cpu().numpy()
    d = datp.shape[1]
    if d == 2:
        plt.scatter(datp[:, 0], datp[:, 1], s=10,
                    alpha=0.5, linewidths=0, c=col)
    elif d == 3:
        ax.scatter(datp[:, 0], datp[:, 1], datp[:, 2],
                   alpha=1, linewidths=0, c=col)
    else:
        # raise NameError("asdf")
        raise Exception("incorrect dimension")
    plt.axis('equal')


def ezshow3D(xyz, col='rgb(0,0,210)', alpha=.2, size=3, show=False):
    trace = Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
                      marker=dict(size=size, color=col, colorscale='Viridis',
                                  opacity=alpha))
    if show:
        layout = Layout(margin=dict(l=0, r=0, b=0, t=0),
                        scene_dragmode='orbit', scene=dict(aspectmode='data'))
        fig = Figure(data=[trace], layout=layout)
        fig.show()
    return trace


class SpecialLosses():
    def __init(self):
        super().__init__()

    def grad_to_jac(grad):
        dim = grad.shape[1]
        return grad[:, 0:dim, 0:dim]

    def radialKE(tz, z_dots):
        dir = tz[:, 1:]
        normalizedRadial = dir/dir.norm(p=2, dim=1, keepdim=True)
        return (z_dots*normalizedRadial).sum(dim=1)**2
    
    def polarKE(tz, z_dots):
        dir = tz[:, 1:]
        normalizedRadial = dir/dir.norm(p=2, dim=1, keepdim=True)
        
        # build A = I - v*v', where v is normalized outwards dir from origin
        A = -torch.bmm(normalizedRadial[:,:,None], normalizedRadial[:,:,None].permute((0, 2, 1)))
        A[:,0,0]+=1
        A[:,1,1]+=1
        
        Azd = torch.bmm(A, z_dots[:,:,None])
        zdAzd = torch.bmm(z_dots[:,None,:],Azd)
        
        return zdAzd.squeeze()

    def jac_to_losses(z_jacs):
        dim = z_jacs.shape[1]
        N = z_jacs.shape[0]

        # divergence squared
        div2loss = torch.zeros(N).to(device)
        for i in range(dim):
            div2loss += z_jacs[:, i, i]
        div2loss = div2loss**2
        # curl
        if dim==2:
            curlvector = z_jacs[:,1,0] - z_jacs[:,0,1]
        else:
            c1 = z_jacs[:,2,1] - z_jacs[:,2,1]
            c2 = z_jacs[:,0,2] - z_jacs[:,2,0]
            c3 = z_jacs[:,1,0] - z_jacs[:,0,1]
            curlvector = torch.stack((c1,c2,c3),axis=1)
        # square norm of curl
        curl2loss = torch.norm(
            z_jacs - z_jacs.transpose(1, 2), p='fro', dim=(1, 2))**2/2
        # pdb.set_trace()
        
        # rigid motion: x(t) -> e^[wt] x0 + kt.
        # v = x_dot = [w]x0+k; dvdx = [w].
        # ==> skew symmetric velocity gradient is rigid.
        # if J is displacement gradient, F=J+I is the deformation gradient,
        # then F'F-I is the green strain.
        # Linearizing this with small J results in J+J'
        rigid2loss = torch.norm(
            z_jacs + z_jacs.transpose(1, 2), p='fro', dim=(1, 2))**2/4
        # v-field gradient loss
        vgradloss = torch.norm(z_jacs, p='fro', dim=(1, 2))**2

        return div2loss, curl2loss, rigid2loss, vgradloss, curlvector


class ImageDataset():
    """Sample from a distribution defined by an image."""

    def __init__(self, imgname, thresh=.2, cannylow=50, cannyhigh=200,
                 rgb_weights=[0.2989, 0.5870, 0.1140, 0], noise_std=.005, binary = True):
        imgrgb = cv.imread(imgname, cv.IMREAD_UNCHANGED)
        img = cv.cvtColor(imgrgb, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(imgrgb, cannylow, cannyhigh)
        self.img = img.copy()
        self.edges = edges.copy()
        imgd = img.astype('float')
        edgesd = edges.astype('float')
        imgd/=imgd.max()
        imgd[imgd >= thresh] = 1 # chop off near whites become white.
        if binary:
            imgd[imgd < thresh] = 0
        
            
        imgd = 1-imgd
        h1, w1 = imgd.shape

        MAX_VAL = .5
        xx = np.linspace(-MAX_VAL, MAX_VAL, w1)
        yy = np.linspace(-MAX_VAL, MAX_VAL, h1)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        self.means = np.concatenate([xx, yy], 1)

        self.probs = imgd.reshape(-1)
        self.probs /= self.probs.sum()
        self.silprobs = edgesd.reshape(-1)
        self.silprobs /= self.silprobs.sum()

        self.noise_std = noise_std

    def sample(self, n_inner=500, n_sil=500, scale=[1, -1], center=[0, 0], rotate=0.):
        rotate = torch.tensor(rotate)
        s, c = (torch.sin(rotate), torch.cos(rotate))
        rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        
        samps = torch.zeros((0,2))
        silsamps = torch.zeros((0,2))
        
        if n_inner!=0:
            inds = np.random.choice(
                int(self.probs.shape[0]), int(n_inner), p=self.probs)
            m = self.means[inds]
            samps = torch.matmul(torch.from_numpy(m).type(torch.FloatTensor), rot) * torch.tensor(scale) + torch.tensor(center)

        if n_sil!=0:
            sinds = np.random.choice(
                int(self.silprobs.shape[0]), int(n_sil), p=self.silprobs)
            ms = self.means[sinds]
            silsamples = np.random.randn(*ms.shape) * self.noise_std + ms
            silsamps = torch.matmul(torch.from_numpy(silsamples).type(torch.FloatTensor), rot) * torch.tensor(scale) + torch.tensor(center)

        # pdb.set_trace()
        return samps, silsamps

    def make_image(n=10000):
        """Make an X shape."""
        points = np.zeros((n, 2))
        points[:n//2, 0] = np.linspace(-1, 1, n//2)
        points[:n//2, 1] = np.linspace(1, -1, n//2)
        points[n//2:, 0] = np.linspace(1, -1, n//2)
        points[n//2:, 1] = np.linspace(1, -1, n//2)
        np.random.seed(42)
        noise = np.clip(np.random.normal(
            scale=0.1, size=points.shape), -0.2, 0.2)
        np.random.seed(None)
        points += noise
        img, _ = np.histogramdd(points, bins=40, range=[
                                [-1.5, 1.5], [-1.5, 1.5]])
        return img

    def normalize_samples(z_target):
        # normalize a [K,N,D] tensor.
        # K is number of frames. N is number of samples. D is dimension.
        # Fit into [-1,1] box without changing aspect ratio.
        # centered on tight bounding box center.
        BB0 = BoundingBox(z_target)
        z_target -= BB0.C
        BB1 = BoundingBox(z_target)
        z_target /= max(BB1.mac)
        z_target /= 1.1  # adds buffer to the keyframes from -1,1 border.

        def transform(x): return (x - BB0.C) / max(BB1.mac) / 1.1

        return z_target, transform


class MeshDataset():
    def __init__(self, mesh_file):
        self.mesh = trimesh.load(mesh_file)
        self.mesh_file = mesh_file
        self.useCache = False # had a point cache when volume point sampling was painfully slow. now that its based on tet meshing, its super fast and the cache isnt needed.
        
        # tet mesh
        mesh_info = MeshInfo()
        mesh_info.set_points(self.mesh.vertices)
        mesh_info.set_facets(self.mesh.faces)
        tetmesh = build(mesh_info)
        
        TV = np.array([val for val in tetmesh.points])
        TT = np.array([val for val in tetmesh.elements])
        
        self.TV = TV;
        self.TT = TT;
        v1 = TV[TT[:,0],:]
        v2 = TV[TT[:,1],:]
        v3 = TV[TT[:,2],:]
        v4 = TV[TT[:,3],:]
        self.tetVols = np.sum(np.cross(v2-v1, v3-v1)*(v4-v1),axis=1);
        self.tetv1=v1;
        self.tetv2=v2;
        self.tetv3=v3;
        self.tetv4=v4;
        
    def getCacheName(mesh_file):
        rname, ext = os.path.splitext(mesh_file)
        fname = os.path.basename(rname)
        dname = os.path.dirname(rname)
        return os.path.join(dname, f".{fname}_pointstore") + ext

    def clearCache(self):
        cacheName = MeshDataset.getCacheName(self.mesh_file)
        if os.path.exists(cacheName):
            os.remove(cacheName)

    # saves/loads sampled points
    def sample(self, n_inner=70, n_surface=30, combined=False):
        if self.useCache:
            # load cache. check for already sampled points.
            cacheName = MeshDataset.getCacheName(self.mesh_file)
            if os.path.exists(cacheName):
                cdict = torch.load(cacheName)
            else:
                cdict = {'pts_inner': np.empty(
                    (0, 3)), 'pts_surface': np.empty((0, 3))}
            old_pts_inner = cdict["pts_inner"]
            old_pts_surface = cdict["pts_surface"]

            # draw point samples to fill cache
            n_new_pts_inner = max(n_inner - old_pts_inner.shape[0], 0)
            n_new_pts_surface = max(n_surface - old_pts_surface.shape[0], 0)
            new_pts_inner, new_pts_surface = self.sample_new(
                n_inner=n_new_pts_inner, n_surface=n_new_pts_surface)

            # save cache
            pts_inner = np.append(old_pts_inner, new_pts_inner, axis=0)
            pts_surface = np.append(old_pts_surface, new_pts_surface, axis=0)
            if n_new_pts_inner != 0 or n_new_pts_surface != 0:
                cdict = {'pts_inner': pts_inner, 'pts_surface': pts_surface}
                torch.save(cdict, cacheName)

            # draw points needed from cache
            subsample_inds_inner = torch.randperm(pts_inner.shape[0])[:n_inner]
            subsample_inds_surface = torch.randperm(
                pts_surface.shape[0])[:n_surface]

            inner_toreturn = pts_inner[subsample_inds_inner, :]
            surface_toreturn = pts_surface[subsample_inds_surface, :]
        else:
            inner_toreturn, surface_toreturn = self.sample_new(
                n_inner=n_inner, n_surface=n_surface)
                
        if not combined:
            return inner_toreturn, surface_toreturn
        else:
            return np.append(inner_toreturn, surface_toreturn, axis=0)

    def sample_new(self, n_inner=70, n_surface=30):
        pts_surface, _ = trimesh.sample.sample_surface(self.mesh, n_surface)
        pts_inner = self.sample_volume(n_inner)

        return pts_inner, pts_surface
    def sample_volume(self, num=100):
        ntets = self.TT.shape[0];
        selected = np.random.choice(ntets, size=(num), replace=True, p=self.tetVols/sum(self.tetVols))
    
        w = -np.log(np.random.rand(num,4))
        w = w/np.sum(w,axis=1,keepdims=True)
        
        volume_pts = (w[:,0:1]*self.tetv1[selected,:]) + (w[:,1:2]*self.tetv2[selected,:]) + (w[:,2:3]*self.tetv3[selected,:]) + (w[:,3:4]*self.tetv4[selected,:])
        
        return volume_pts
    def plotly_trace(self, color=None, opacity=.8):
        X = self.mesh.vertices
        T = self.mesh.faces
        fc = None

        if color is None:
            fc = (X[T[:, 0], :]+X[T[:, 1], :]+X[T[:, 2], :])/3
        gob = go.Mesh3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], i=T[:, 0], j=T[:, 1],
                        k=T[:, 2], color=color, opacity=opacity, facecolor=fc,
                        flatshading=True)
        return gob

    def meshArrayToTraces(meshArray, color=None, opacity=.8, show=False):
        traces = []

        if color is None:
            color = []
            cs = np.array([1, .1, .1])  # start color
            cf = np.array([1, .64, .2])  # end color
            colorinterp = np.linspace(0, 1, len(meshArray))
            for i in range(len(meshArray)):
                ct = np.round(((1-colorinterp[i])*cs + cf*colorinterp[i])*255)
                color.append(f"rgb({ct[0]:03},{ct[1]:03},{ct[2]:03})")

        for i in range(len(meshArray)):
            if type(color) is list:
                col = color[0]
                if len(color) > 1:
                    col = color[i]
            else:
                col = color
            traces.append(meshArray[i].plotly_trace(
                color=col, opacity=opacity))

        if show:
            layout = Layout(margin=dict(l=0, r=0, b=0, t=0),
                            scene_dragmode='orbit',
                            scene=dict(aspectmode='data'))
            Figure(data=traces, layout=layout).show()

        return traces

    def meshArrayToPoints(meshArray, inner_percentage, n_total, combined=True):

        n_inner = round(n_total*inner_percentage)
        n_surface = n_total - n_inner
        pts_inner = np.zeros((len(meshArray), n_inner, 3))
        pts_surface = np.zeros((len(meshArray), n_surface, 3))
        for i in range(len(meshArray)):
            pts_inner[i, :, :], pts_surface[i, :, :] = meshArray[i].sample(
                n_inner=n_inner, n_surface=n_surface, combined=False)

        if combined:
            return np.concatenate((pts_inner, pts_surface), axis=1) 

        return pts_inner, pts_surface


class BoundingBox():
    # use like:
    # BB = BoundingBox(z_target);
    # smps = BB.sampleuniform(t_N = 30, x_N = 10, y_N = 11, z_N=12, bbscale = 1.1);
    # smps = BB.samplerandom(N = 10000, bbscale = 1.1);

    def __init__(self, z_target_full, square=False):
        self.T = z_target_full.shape[0]
        self.dim = z_target_full.shape[2]

        # min corner, max corner, center
        self.mic = z_target_full.reshape(-1, self.dim).min(0)[0].detach()
        self.mac = z_target_full.reshape(-1, self.dim).max(0)[0].detach()
        self.C = (self.mic+self.mac)/2

        if square:
            # min corner, max corner, center
            self.mac = self.C + (self.mac - self.C).max()
            self.mic = self.C - (self.mac - self.C).max()

    def extendedBB(self, bbscale=1.1, returnNP=False):
        # extended bounding box.
        emic = ((self.mic-self.C)*bbscale+self.C)
        emac = ((self.mac-self.C)*bbscale+self.C)

        if returnNP:
            return emic.cpu().detach().numpy(), emac.cpu().detach().numpy()

        return emic, emac

    def sampleuniform(self, t_N=30, x_N=10, y_N=11, z_N=12, bbscale=1.1):
        [eLL, eTR] = self.extendedBB(bbscale)

        tspace = torch.linspace(0, self.T-1, t_N)
        xspace = torch.linspace(eLL[0], eTR[0], x_N)
        yspace = torch.linspace(eLL[1], eTR[1], y_N)
        if self.dim == 3:
            zspace = torch.linspace(eLL[2], eTR[2], z_N)
            xgrid, ygrid, zgrid, tgrid = torch.meshgrid(
                xspace, yspace, zspace, tspace, indexing='ij')
            z_sample = torch.transpose(torch.reshape(torch.stack(
                [tgrid, xgrid, ygrid, zgrid]), (4, -1)), 0, 1).to(device)
        else:
            xgrid, ygrid, tgrid = torch.meshgrid(
                xspace, yspace, tspace, indexing='ij')
            z_sample = torch.transpose(torch.reshape(torch.stack(
                [tgrid, xgrid, ygrid]), (3, -1)), 0, 1).to(device)

        return z_sample.to(device)

    def samplerandom(self, N=10000, bbscale=1.1):
        [eLL, eTR] = self.extendedBB(bbscale)
        # time goes from 0 to T-1
        dT = torch.Tensor([self.T-1]).to(device)  # size of time begin to end
        TC = torch.Tensor([(self.T-1.0)/2.0]).to(device)  # time center

        z_sample = torch.rand(N, self.dim + 1).to(device)-0.5
        deltx = torch.cat((dT, eTR-eLL))
        z_sample = deltx*z_sample + torch.cat((TC, self.C))

        return z_sample


class InputMapping(nn.Module):
    """Fourier features mapping"""

    def __init__(self, d_in, n_freq, sigma=2, tdiv=2, incrementalMask=True, Tperiod=None):
        super().__init__()
        Bmat = torch.randn(n_freq, d_in) * np.pi* sigma/np.sqrt(d_in)  # gaussian
        # time frequencies are a quarter of spacial frequencies.
        # Bmat[:, d_in-1] /= tdiv
        Bmat[:, 0] /= tdiv

        self.Tperiod = Tperiod
        if Tperiod is not None:
            # Tcycles = (Bmat[:, d_in-1]*Tperiod/(2*np.pi)).round()
            # K = Tcycles*(2*np.pi)/Tperiod
            # Bmat[:, d_in-1] = K
            Tcycles = (Bmat[:, 0]*Tperiod/(2*np.pi)).round()
            K = Tcycles*(2*np.pi)/Tperiod
            Bmat[:, 0] = K
        
        Bnorms = torch.norm(Bmat, p=2, dim=1)
        sortedBnorms, sortIndices = torch.sort(Bnorms)
        Bmat = Bmat[sortIndices, :]

        self.d_in = d_in
        self.n_freq = n_freq
        self.d_out = n_freq * 2 + d_in if Tperiod is None else n_freq * 2 + d_in - 1
        self.B = nn.Linear(d_in, self.d_out, bias=False)
        with torch.no_grad():
            self.B.weight = nn.Parameter(Bmat.to(device), requires_grad=False)
            self.mask = nn.Parameter(torch.zeros(
                1, n_freq), requires_grad=False)

        self.incrementalMask = incrementalMask
        if not incrementalMask:
            self.mask = nn.Parameter(torch.ones(
                1, n_freq), requires_grad=False)

    def step(self, progressPercent):
        if self.incrementalMask:
            float_filled = (progressPercent*self.n_freq)/.7
            int_filled = int(float_filled // 1)
            remainder = float_filled % 1

            if int_filled >= self.n_freq:
                self.mask[0, :] = 1
            else:
                self.mask[0, 0:int_filled] = 1
                # self.mask[0, int_filled] = remainder

    def forward(self, xi):
        # pdb.set_trace()
        dim = xi.shape[1]-1
        y = self.B(xi)
        if self.Tperiod is None:
            return torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi], dim=-1)
        else:
            return torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi[:,1:dim+1]], dim=-1)


class SaveTrajectory():

    def gpu_usage(devnum=0):
        allocated = round(torch.cuda.memory_allocated(devnum)/1024**3, 2)
        reserved = round(torch.cuda.memory_reserved(devnum)/1024**3, 2)
        print('Allocated:', allocated, 'GB', ' Reserved:', reserved, 'GB')

    def save_losses(losses_in, separate_losses_in,
                    outfolder='results/outcache/', savename='losses.pdf',
                    start=1, end=10000, maxcap=100):
        # SEPARATE LOSSES PLOT
        losses = losses_in.copy()
        separate_losses = separate_losses_in.copy()
        separate_losses[separate_losses > maxcap] = maxcap
        losses[losses > maxcap] = maxcap
        (fig, (ax1, ax2)) = plt.subplots(2, 1)
        ax1.plot(losses[0, start:end], 'k')
        ax1.set_ylabel(f'loss\n{losses[0,:].min().item():.2f}')
        ax1.set_yscale("log")
        ax2.plot(separate_losses[0, start:end], 'g')
        ax2.plot(separate_losses[1, start:end], 'g')
        # ax2.plot(separate_losses[0,start:end]*100,'g');
        # ax2.plot(separate_losses[1,start:end]*100,'g');
        ax2.plot(separate_losses[6, start:end], 'y')  # self adv
        ax2.plot(separate_losses[7, start:end], 'c')  # accel
        ax2.plot(separate_losses[9, start:end], 'r')  # kurv
        ax2.plot(separate_losses[12, start:end], 'b')  # u div
        # ax2.plot(separate_losses[2,start:end],'k');
        # ax2.plot(separate_losses[4,start:end],'k');
        # ax2.plot(separate_losses[5,start:end],'k');
        # ax2.plot(separate_losses[6,start:end],'k');
        # ax2.plot(separate_losses[7,start:end],'k');
        # ax2.plot(separate_losses[8,start:end],'k');
        # ax2.plot(separate_losses[11,start:end],'k');
        # ax2.plot(separate_losses[12,start:end],'k');
        ax2.set_ylabel('loss')
        plt.savefig(outfolder + savename)

    def save_trajectory(model, z_target_full, savedir='results/outcache/',
                        savename='', nsteps=20, dpiv=100, n=4000, alpha=.5,
                        ot_type=2, meshArray=None,
                        rbf=True, sigma=None, knn=20, opt=False, reach=None):
        # handler for different dimensions
        if z_target_full.shape[2] == 2:
            return SaveTrajectory.save_trajectory_2d(model, z_target_full, savedir,
                                              savename, nsteps, dpiv, n, alpha,
                                              ot_type, reach=reach)
        else:
            return SaveTrajectory.save_trajectory_3d(model, z_target_full, savedir,
                                              savename, nsteps, dpiv, n, alpha,
                                              ot_type, meshArray=meshArray, reach=reach)

    def save_trajectory_2d(model, z_target_full, savedir='results/outcache/',
                           savename='', nsteps=20, dpiv=100, n=4000, alpha=.5,
                           ot_type=2, reach=None):
        z_target_full = z_target_full.detach()

        with torch.no_grad():
            # save model
            if not os.path.exists(savedir+'models/'):
                os.makedirs(savedir+'models/')
            model.save_state(fn=savedir + 'models/state_' + savename + '.tar')

            # save trajectory video0
            if n > z_target_full.shape[1]:
                n = z_target_full.shape[1]
            subsample_inds = torch.randperm(z_target_full.shape[1])[:n]
            z_target = z_target_full[:, subsample_inds, :]

            T = z_target.shape[0]
            integration_times = torch.linspace(0, T-1, nsteps).to(device)
            x_traj_reverse_t = model(
                z_target[T-1, :, :], integration_times, reverse=True)
            x_traj_forward_t = model(
                z_target[0, :, :], integration_times, reverse=False)
            x_traj_reverse = x_traj_reverse_t.detach().cpu().numpy()
            x_traj_forward = x_traj_forward_t.detach().cpu().numpy()

            allpoints = torch.cat(
                (x_traj_reverse_t, x_traj_forward_t, z_target), dim=0)
            BB = BoundingBox(allpoints, square=False)
            emic, emac = BB.extendedBB(1.1)
            z_sample = BB.sampleuniform(t_N=1, x_N=20, y_N=20)
            z_sample_d = z_sample.detach().cpu().numpy()
            fig, (ax) = plt.subplots(1, 1)
            
            # forward
            moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
            with moviewriter.saving(fig, savedir+'forward_'+savename+'.mp4', dpiv):
                for i in range(nsteps):
                    for t in range(T):
                        plt.scatter(
                            z_target.detach().cpu().numpy()[t, :, 0],
                            z_target.detach().cpu().numpy()[t, :, 1],
                            s=10, alpha=alpha, linewidths=0, c='green',
                            edgecolors='black')
                    x_traj = x_traj_forward

                    # plot velocities
                    z_dots_d = model.velfunc.get_z_dot(
                        z_sample[:, 0]*0.0 + integration_times[i],
                        z_sample[:, 1:]).detach().cpu().numpy()
                    plt.quiver(z_sample_d[:, 1], z_sample_d[:, 2],
                               z_dots_d[:, 0], z_dots_d[:, 1])
                    plt.scatter(x_traj[i, :, 0], x_traj[i, :, 1], s=10,
                                alpha=alpha, linewidths=0, c='blue',
                                edgecolors='black')

                    ax.axis('equal')
                    plt.axis('equal')
                    ax.set(xlim=(emic[0].item(), emac[0].item()),
                           ylim=(emic[1].item(), emac[1].item()))
                    plt.axis('off')
                    moviewriter.grab_frame()
                    plt.clf()
                moviewriter.finish()

            # reverse
            moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
            with moviewriter.saving(fig, savedir+'rev_'+savename+'.mp4', dpiv):
                for i in range(nsteps):
                    for t in range(T):
                        plt.scatter(
                            z_target.detach().cpu().numpy()[t, :, 0],
                            z_target.detach().cpu().numpy()[t, :, 1],
                            s=10, alpha=alpha, linewidths=0, c='green',
                            edgecolors='black')
                    x_traj = x_traj_reverse

                    # plot velocities
                    z_dots_d = model.velfunc.get_z_dot(
                        z_sample[:, 0]*0.0 + integration_times[(nsteps-1)-i],
                        z_sample[:, 1:]).detach().cpu().numpy()
                    plt.quiver(z_sample_d[:, 1],
                               z_sample_d[:, 2], -z_dots_d[:, 0], -z_dots_d[:, 1])
                    plt.scatter(x_traj[i, :, 0], x_traj[i, :, 1], s=10,
                                alpha=alpha, linewidths=0, c='blue',
                                edgecolors='black')

                    ax.axis('equal')
                    plt.axis('equal')
                    ax.set(xlim=(emic[0].item(), emac[0].item()),
                           ylim=(emic[1].item(), emac[1].item()))
                    plt.axis('off')
                    moviewriter.grab_frame()
                    plt.clf()
                moviewriter.finish()

            # forward and back
            ts = torch.linspace(0, 1, nsteps)
            moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
            x_trajs = torch.zeros(n, 2, (T-1)*(nsteps-1)+1)
            t_trajs = torch.zeros((T-1)*(nsteps-1)+1)
            trajsc = 0
            indices = torch.arange(0, z_target.shape[1])
            with moviewriter.saving(fig, savedir+'fb_'+savename+'.mp4', dpiv):
                for tt in range(T-1):
                    if tt > 0:
                        # this permutation is needed to keep x_trajs continuous. otherwise at keyframes, the permutation gets reset.
                        _fst, indices = MiscTransforms.OT_registration_POT_2D(
                            x_traj_t, z_target[tt, :, :])
                    integration_times = torch.linspace(
                        tt, tt+1, nsteps).to(device)
                    x_traj_reverse_t = model(
                        z_target[tt+1, :, :], integration_times, reverse=True)
                    x_traj_forward_t = model(
                        z_target[tt, indices, :], integration_times, reverse=False)
                    # x_traj_reverse = x_traj_reverse_t.detach().cpu().numpy()
                    # x_traj_forward = x_traj_forward_t.detach().cpu().numpy()

                    endstep = nsteps if tt == T-2 else nsteps-1
                    init = None
                    for i in range(endstep):
                        fs = x_traj_forward_t[i, :, :]
                        ft = x_traj_reverse_t[(nsteps-1)-i, :, :]

                        # ground truth keyframes
                        for t in range(T):
                            plt.scatter(z_target.detach().cpu().numpy()[t, :, 0],
                                        z_target.detach().cpu().numpy()[t, :, 1],
                                        s=10, alpha=alpha, linewidths=0, c='green',
                                        edgecolors='black')

                        # plot velocities
                        z_dots_d = model.velfunc.get_z_dot(
                            z_sample[:, 0]*0.0 + integration_times[i],
                            z_sample[:, 1:]).detach().cpu().numpy()
                        plt.quiver(z_sample_d[:, 1], z_sample_d[:, 2],
                                   z_dots_d[:, 0], z_dots_d[:, 1], lw=.01)

                        # forward and backwards separately
                        fsp = fs.detach().cpu().numpy()
                        ftp = ft.detach().cpu().numpy()
                        plt.scatter(fsp[:, 0], fsp[:, 1], s=10, alpha=alpha,
                                    linewidths=0, c='yellow', edgecolors='black')
                        plt.scatter(ftp[:, 0], ftp[:, 1], s=10, alpha=alpha,
                                    linewidths=0, c='orange', edgecolors='black')

                        if reach is None:
                            # W2 barycenter combination
                            if ot_type == 1:
                                # this registration isn't 1-1 on point clouds. scaling parameter needs to be high enough to get 1-1.
                                fst = MiscTransforms.OT_registration(fs, ft)
                            elif ot_type == 2:
                                # full linear program version of OT. slightly slower than geomloss but frankly not that slow compared to other steps in the pipeline.
                                fst, indices = MiscTransforms.OT_registration_POT_2D(
                                    fs, ft)

                            x_traj_t = (fs*(1-ts[i]) + fst*ts[i])
                        else:
                            
                            ubc_dir_name = savedir+'unbalanced_convergence'
                            if not os.path.exists(ubc_dir_name):
                                os.makedirs(ubc_dir_name)
                            x_traj_t = MiscTransforms.unbalanced_OT_Barycenter(fs, ft, ts[i],reach,init,tag=ubc_dir_name+'/'+savename+"_"+str(tt)+"_"+str(ts[i].item()))
                    
                        x_traj = x_traj_t.detach().cpu().numpy()
                        plt.scatter(x_traj[:, 0], x_traj[:, 1], s=10, alpha=alpha,
                                    linewidths=0, c='blue', edgecolors='black')

                        x_trajs[:, :, trajsc] = x_traj_t
                        t_trajs[trajsc] = integration_times[i]
                        trajsc += 1

                        ax.axis('equal')
                        plt.axis('equal')
                        ax.set(xlim=(emic[0].item(), emac[0].item()),
                               ylim=(emic[1].item(), emac[1].item()))
                        plt.axis('off')
                        moviewriter.grab_frame()
                        plt.clf()
                moviewriter.finish()
            plt.close(fig)
            
        # save points as nframes, npoints, dim
        np.save(savedir+'x_trajs_'+savename+'.npy', x_trajs.permute((2,0,1)).detach().numpy())
            
        return x_trajs, t_trajs, nsteps, T
    
    def get_cubic_OT_trajectory(z_target_full, nsteps=20, n=4000, savedir="results/outcache", savename = ""):
        z_target_full = z_target_full.detach()

        with torch.no_grad():
            # subsample.
            if n > z_target_full.shape[1]:
                n = z_target_full.shape[1]
            subsample_inds = torch.randperm(z_target_full.shape[1])[:n]
            z_target = z_target_full[:, subsample_inds, :]

            # compute consecutive OT mappings between keyframes
            T = z_target.shape[0]            
            for tt in range(T-1):
                _fst, _indices = MiscTransforms.OT_registration_POT_2D(z_target[tt, :, :], z_target[tt+1, :, :])
                z_target[tt+1, :, :] = _fst
            
            x_trajs = torch.zeros(n, 2, (T-1)*(nsteps-1)+1)
            t_trajs = torch.zeros((T-1)*(nsteps-1)+1)
            
            # build cubic splines
            for i in range(n):
                x = torch.arange(T)
                y = z_target[:,i,:].cpu()
                cs = scipyinterpolate.CubicSpline(x,y,axis=0)
                ys = cs(torch.linspace(0,T-1, (T-1)*(nsteps-1)+1))
                x_trajs[i,:,:] = torch.tensor(ys).t()
                
        # save points as nframes, npoints, dim
        np.save(savedir+'x_trajs_'+savename+'.npy', x_trajs.permute((2,0,1)).detach().numpy())
            
        return x_trajs, t_trajs, nsteps, T
    
    
    # get the piecewise W2 interpolation between keyframes. Like waddintonOT, or if the model only performed identity maps.
    def get_OT_trajectory(z_target_full, nsteps=20, n=4000, ot_type=2, savedir='results/outcache/', savename=''):
        z_target_full = z_target_full.detach()

        with torch.no_grad():
            # save trajectory video0
            if n > z_target_full.shape[1]:
                n = z_target_full.shape[1]
            subsample_inds = torch.randperm(z_target_full.shape[1])[:n]
            z_target = z_target_full[:, subsample_inds, :]

            T = z_target.shape[0]            
            # forward and back
            ts = torch.linspace(0, 1, nsteps)
            x_trajs = torch.zeros(n, 2, (T-1)*(nsteps-1)+1)
            t_trajs = torch.zeros((T-1)*(nsteps-1)+1)
            trajsc = 0
            indices = torch.arange(0, z_target.shape[1])
            for tt in range(T-1):
                if tt > 0:
                    # this permutation is needed to keep x_trajs continuous. otherwise at keyframes, the permutation gets reset.
                    _fst, indices = MiscTransforms.OT_registration_POT_2D(
                        x_traj_t, z_target[tt, :, :])
                
                integration_times = torch.linspace(
                    tt, tt+1, nsteps).to(device)
                
                fs = z_target[tt, :, :]
                ft = z_target[tt+1, :, :]
                
                # W2 barycenter combination
                if ot_type == 1:
                    # this registration isn't 1-1 on point clouds. don't know why currently.
                    fst = MiscTransforms.OT_registration(fs, ft)
                elif ot_type == 2:
                    # full linear program version of OT. slightly slower than geomloss but frankly not that slow compared to other steps in the pipeline.
                    fst, indices = MiscTransforms.OT_registration_POT_2D(
                        fs, ft)

                endstep = nsteps if tt == T-2 else nsteps-1
                for i in range(endstep):
                    x_traj_t = (fs*(1-ts[i]) + fst*ts[i])
                    x_trajs[:, :, trajsc] = x_traj_t
                    t_trajs[trajsc] = integration_times[i]
                    trajsc += 1
        
        # save points as nframes, npoints, dim
        np.save(savedir+'x_trajs_'+savename+'.npy', x_trajs.permute((2,0,1)).detach().numpy())
        
        return x_trajs, t_trajs, nsteps, T
    
    def render_2d(model, z_target_full, xt_trajs, 
                  savedir='results/outcache/', savename='', dpiv=600, 
                  sigma=None, knn=20, cycle=False, lw = .5, contrast = 3, keyframes = True, Nqvr = 150, Nrbf=10000, showVelocity = True, plotKeypoints=False,tightBB=True):
        x_trajs, t_trajs, nsteps, T = xt_trajs
        dim = x_trajs.shape[1]
        assert z_target_full.shape[0]==T
        assert x_trajs.shape[2] == (T-1)*(nsteps-1)+1
        
        imsavefolder = savedir + 'traj_pics/'
        if not os.path.exists(imsavefolder):
            os.makedirs(imsavefolder)
        
        with torch.no_grad():
            # render
            fig, (ax) = plt.subplots(1, 1)
            n = x_trajs.shape[0]  # num particles
            nf = x_trajs.shape[2]  # number of frames in full trajectory
            nft = torch.linspace(0, 1, nf)  # color tracers
            cs = torch.tensor((.3, .5, 1))  # start color
            cf = torch.tensor((.2, 1, .2))  # end color
            x_trajs_f = x_trajs.transpose(1, 2)
            moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
            fig.tight_layout()
            ax.axis('equal')

            # set up bounding box and uniform quiver locations
            full_traj = BoundingBox(x_trajs[:, :, :].permute((2,0,1)), square=False)
            emic, emac = full_traj.extendedBB(1.1)
            width=emac[0].item()-emic[0].item()
            height=emac[1].item()-emic[1].item()
            widthSamples = width/height
            nH = int(np.floor(np.sqrt(Nqvr*height/width)))
            nW = int(np.floor(Nqvr/nH))
            z_sample = full_traj.sampleuniform(t_N=1, x_N=nW, y_N=nH)
            z_sample_d = z_sample.cpu().numpy()

            # get largest single BB width and height that covers all frame rbfs individually
            frameBB = BoundingBox(x_trajs[:, :, 0:1].permute((2,0,1)), square=False)
            emicM, emacM = frameBB.extendedBB(1.2); wM = emacM[0]-emicM[0]; hM = emacM[1]-emicM[1]
            for t in range(0,nf):
                frameBB = BoundingBox(x_trajs[:, :, t:t+1].permute((2,0,1)), square=False)
                emicT, emacT = frameBB.extendedBB(1.2); wT = emacT[0]-emicT[0]; hT = emacT[1]-emicT[1]
                wM = wM if wM > wT else wT
                hM = hM if hM > hT else hT
            
            ax.axis('off')
            plt.scatter([emic[0].item(), emac[0].item()], [emic[1].item(), emac[1].item()], alpha=0, linewidths=0)
            dullingfactor = .6
            with moviewriter.saving(fig, savedir + 'traj_'+savename+'.mp4',
                                    dpiv):
                keyframe_percentage_curr = -1
                for t in range(0, nf):
                    c_interp = nft[t]
                    if cycle:
                        c_interp = 1-2*abs(nft[t]-.5)
                    ctt =  (cs*(1-c_interp) + cf*c_interp)
                    
                    dctt = ctt*dullingfactor
                    ct = (ctt[0].item(), ctt[1].item(), ctt[2].item())
                    dct = (dctt[0].item(), dctt[1].item(), dctt[2].item())

                    # plot velocities
                    if showVelocity:
                        z_dots_d = model.velfunc.get_z_dot(
                            z_sample[:, 0]*0.0 + t_trajs[t],
                            z_sample[:, 1:]).cpu().numpy()
                        qvr = ax.quiver(z_sample_d[:, 1], z_sample_d[:, 2],
                                        z_dots_d[:, 0], z_dots_d[:, 1],
                                        headwidth=1, headlength=3,
                                        headaxislength=2,zorder=5)

                    # plot keyframes as tracers pass by
                    dontremovescr = False
                    keyframe_percentage = np.floor(t/(nf-1.)*(T-1))
                    if keyframe_percentage != keyframe_percentage_curr:
                        keyframe_percentage_curr = keyframe_percentage
                        tt = int(keyframe_percentage)
                        dontremovescr = True
                        
                    if t > 0:
                        segment_t = x_trajs_f[:, t -
                                              1:t+1].cpu().numpy()
                        lc = mc.LineCollection(segment_t, color=ct, lw=lw,
                                               zorder=1)
                        ax.add_collection(lc)

                    # plot endpoints
                    points = x_trajs[:, :, t].to(device)
                    if plotKeypoints:
                        pointsp = points.detach().cpu().numpy()
                        kyp = ax.scatter(pointsp[:,0],pointsp[:,1],s=10, alpha=1,linewidths=0, color=dct, edgecolors='black')
                    
                    frameBB = BoundingBox(x_trajs[:, :, t:t+1].permute((2,0,1)), square=False)
                    emicf, emacf = frameBB.extendedBB(1.2)
                    if not tightBB:
                        # use single precomputed bounding box for all frames.
                        cmf = (emicf + emacf)/2
                        emicf[0] = cmf[0]-wM/2
                        emicf[1] = cmf[1]-hM/2
                        emacf[0] = cmf[0]+wM/2
                        emacf[1] = cmf[1]+hM/2

                    if Nrbf != 0:
                        if sigma is not None:
                            sigmas = torch.tensor(sigma).to(device)
                        else:
                            pdists = torch.tensor(
                                squareform(torch.pdist(points.cpu()))
                            ).to(device)
                            sigmas = pdists.topk(
                                knn+1, largest=False).values[:, -1]

                        # sample Ntot points in a rectangular grid, while being fair to aspect ratio
                        width=emacf[0].item()-emicf[0].item()
                        height=emacf[1].item()-emicf[1].item()
                        widthSamples = width/height
                        nH = int(np.floor(np.sqrt(Nrbf*height/width)))
                        nW = int(np.floor(Nrbf/nH))

                        xs = torch.linspace(
                            emicf[0].item(), emacf[0].item(),
                            nW).to(device)
                        ys = torch.linspace(
                            emicf[1].item(), emacf[1].item(),
                            nH).to(device)
                        grid = torch.stack(torch.meshgrid(xs, ys,
                                                          indexing='xy'),
                                           dim=-1)

                        dists = (grid[:, :, None] -
                                 points[None, None]).norm(p=2, dim=-1)
                        zs = torch.exp(
                            -(dists.pow(2) /
                              (2 * sigmas[None, None]**2))).sum(-1)
                        zs -= zs.min()
                        zs /= zs.max()

                        if contrast==1:
                            # lower contrast. slightly mottled inside.
                            zs = .95*(torch.tanh(4*(zs-.6))+1)/2 - .04
                            zs[zs<0]=0
                        elif contrast==2:
                            # pretty sharp boundaries. more constant inside.
                            zs = (torch.tanh(7*(zs-.5))+1)/2
                        else:
                            # experimental. need even sharper boundaries?
                            zs = (torch.tanh(7.5*(zs-.45))+1)/2

                        zs/=zs.max()
                        zs = zs.cpu().numpy()[:, :, None]
                        color = np.array(dct + (1,))
                        color2 = np.array(dct + (0,))
                        color3 = np.array(ct + (0,))

                        im_whiteback = zs * color + (1-zs) * np.array([1, 1, 1, 0]) # back color is white
                        if lw==0:
                            im = im_whiteback
                        else:
                            # im = zs * color + (1-zs) * color2 # back color is same as front
                            im = zs * color + (1-zs) * color3 # back color matches tracers

                        im = (im * 255).astype(np.uint8)
                        scr = ax.imshow(
                            im, extent=(emicf[0].item(), emacf[0].item(),
                                        emicf[1].item(), emacf[1].item()),
                            origin='lower', zorder=4)
                    
                        # save image alone for later use
                        imsavename = imsavefolder + f'pic_'+savename+ f'_{t:04}.jpg';
                        im2save = (im_whiteback * 255).astype(np.uint8)
                        cv.imwrite(imsavename, np.flipud(im2save[:,:,[2, 1, 0, 3]]))
                    
                    ax.set(xlim=(emic[0].item(), emac[0].item()),
                           ylim=(emic[1].item(), emac[1].item()))
                    moviewriter.grab_frame()
                    if not dontremovescr or not keyframes:
                        if  Nrbf != 0:
                            scr.remove()
                    qvr.remove() if showVelocity else None
                    if plotKeypoints:
                        kyp.remove()
            moviewriter.finish()
            plt.close(fig)

    def save_trajectory_3d(model, z_target_full, savedir='results/outcache/',
                           savename='', nsteps=20, dpiv=100, n=4000, alpha=.2,
                           ot_type=2, writeTracers=False, meshArray=None, reach=None):
        # initialize
        if not os.path.exists(savedir+'models/'):
            os.makedirs(savedir+'models/')
        model.save_state(fn=savedir + 'models/state_' + savename + '.tar')
        dim = z_target_full.shape[2]
        T = z_target_full.shape[0]
        colormap = plotly.express.colors.sequential.Viridis
        keytraces = MeshDataset.meshArrayToTraces(
            meshArray, color='blue', opacity=.05, show=False) if meshArray is not None else []
        layout = Layout(margin=dict(l=0, r=0, b=0, t=0), scene_dragmode='orbit', scene=dict(
            aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))

        # subsample points
        if n > z_target_full.shape[1]:
            n = z_target_full.shape[1]
        subsample_inds = torch.randperm(z_target_full.shape[1])[:n]
        z_target = z_target_full[:, subsample_inds, :]

        # get trajectory
        integration_times = torch.linspace(0, T-1, nsteps).to(device)
        x_traj_reverse_t = model(
            z_target[T-1, :, :], integration_times, reverse=True)
        x_traj_forward_t = model(
            z_target[0, :, :], integration_times, reverse=False)
        x_traj_reverse = x_traj_reverse_t.cpu().detach().numpy()
        x_traj_forward = x_traj_forward_t.cpu().detach().numpy()

        # get bounding box of trajectory
        BB = BoundingBox(torch.cat(
            (x_traj_reverse_t, x_traj_forward_t, z_target), dim=0).detach(), square=False)
        emic, emac = BB.extendedBB(1.1, returnNP=True)
        z_sample = BB.sampleuniform(t_N=1, x_N=20, y_N=20)
        z_sample_d = z_sample.cpu().detach().numpy()
        BB_trace = Scatter3d(name="", visible=True, showlegend=False, opacity=0, hoverinfo='none', x=[
                             emic[0], emac[0]], y=[emic[1], emac[1]], z=[emic[2], emac[2]])

        # FORWARD
        nframes = x_traj_forward_t.shape[0]
        xyz = x_traj_forward_t.reshape((nframes*n, dim)).cpu().detach().numpy()
        framenum = torch.repeat_interleave(
            torch.arange(nframes), n).cpu().detach().numpy()
        df = pd.DataFrame(dict(xp=xyz[:, 0], yp=xyz[:, 1], zp=xyz[:, 2],
                               framenum=framenum, size=1, colors=nframes-framenum))
        # plot animation
        fig = px.scatter_3d(df, x="xp", y="yp", z="zp", opacity=alpha,
                            animation_frame="framenum",  # symbol = "framenum",
                            size="size", size_max=10,
                            color="colors", color_continuous_scale=colormap, range_color=[-nframes*1, nframes*.9])
        # remove marker outlines from animation
        for i in range(len(fig.frames)):
            fig.frames[i].data[0]['marker']['line'] = dict(
                width=0, color='DarkSlateGrey')
        # plot background
        for i in range(len(keytraces)):
            fig.add_trace(keytraces[i])
        fig.add_trace(BB_trace)
        fig.update_layout(layout)
        plotly.offline.plot(fig, filename=savedir+'forward_'+savename+'.html')

        # REVERSE
        nframes = x_traj_reverse_t.shape[0]
        xyz = x_traj_reverse_t.reshape((nframes*n, dim)).cpu().detach().numpy()
        framenum = torch.repeat_interleave(
            torch.arange(nframes), n).cpu().detach().numpy()
        df = pd.DataFrame(dict(xp=xyz[:, 0], yp=xyz[:, 1], zp=xyz[:, 2],
                               framenum=framenum, size=1, colors=nframes-framenum))
        # plot animation
        fig = px.scatter_3d(df, x="xp", y="yp", z="zp", opacity=alpha,
                            animation_frame="framenum",  # symbol = "framenum",
                            size="size", size_max=10,
                            color="colors", color_continuous_scale=colormap, range_color=[-nframes*1, nframes*.9])
        # remove marker outlines from animation
        for i in range(len(fig.frames)):
            fig.frames[i].data[0]['marker']['line'] = dict(
                width=0, color='DarkSlateGrey')
        # plot background
        for i in range(len(keytraces)):
            fig.add_trace(keytraces[i])
        fig.add_trace(BB_trace)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene_dragmode='orbit', scene=dict(
            aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
        plotly.offline.plot(fig, filename=savedir+'rev_'+savename+'.html')

        # FORWARD AND BACK
        # initialize variables for trajectory
        ts = torch.linspace(0, 1, nsteps)
        x_trajs = torch.zeros(n, dim, (T-1)*(nsteps-1)+1)
        t_trajs = torch.zeros((T-1)*(nsteps-1)+1)
        trajsc = 0
        indices = torch.arange(0, z_target.shape[1])
        # initialize variables for plotly
        xyz = np.zeros((0, dim))
        framenum = np.zeros(0)
        frame_counter = 0
        colorgroup = np.zeros(0)
        for tt in range(T-1):
            if tt > 0:
                _fst, indices = MiscTransforms.OT_registration_POT_2D(
                    x_traj_t.detach(), z_target[tt, :, :].detach())
            integration_times = torch.linspace(tt, tt+1, nsteps).to(device)
            x_traj_reverse_t = model(
                z_target[tt+1, :, :], integration_times, reverse=True)
            x_traj_forward_t = model(
                z_target[tt, indices, :], integration_times, reverse=False)
            x_traj_reverse = x_traj_reverse_t.cpu().detach().numpy()
            x_traj_forward = x_traj_forward_t.cpu().detach().numpy()

            endstep = nsteps if tt == T-2 else nsteps-1
            init = None
            for i in range(endstep):
                fs = x_traj_forward_t[i, :, :]
                ft = x_traj_reverse_t[(nsteps-1)-i, :, :]
                fsp = fs.cpu().detach().numpy()
                ftp = ft.cpu().detach().numpy()

                
                
                if reach is None:
                    # W2 barycenter combination
                    if ot_type == 1:
                        # this registration isn't always 1-1 on point clouds.
                        fst = MiscTransforms.OT_registration(
                            fs.detach(), ft.detach())
                    elif ot_type == 2:
                        # full linear program version of OT. slightly slower than geomloss but frankly not that slow compared to other steps in the pipeline.
                        fst, indices = MiscTransforms.OT_registration_POT_2D(
                            fs.detach(), ft.detach())

                    x_traj_t = (fs*(1-ts[i]) + fst*ts[i])
                else:
                    ubc_dir_name = savedir+'unbalanced_convergence'
                    if not os.path.exists(ubc_dir_name):
                        os.makedirs(ubc_dir_name)
                    x_traj_t = MiscTransforms.unbalanced_OT_Barycenter(fs, ft, ts[i],reach,init,tag=ubc_dir_name+'/'+savename+"_"+str(tt)+"_"+str(ts[i].item()))
                
                x_traj = x_traj_t.cpu().detach().numpy()

                x_trajs[:, :, trajsc] = x_traj_t
                t_trajs[trajsc] = integration_times[i]
                trajsc += 1

                # record into arrays for plotly
                xyz = np.concatenate((xyz, fsp), axis=0)
                xyz = np.concatenate((xyz, ftp), axis=0)
                xyz = np.concatenate((xyz, x_traj), axis=0)
                framenum = np.concatenate((framenum, np.ones(
                    fs.shape[0]+ft.shape[0]+x_traj.shape[0])*frame_counter), axis=0)
                colorgroup = np.concatenate(
                    (colorgroup, np.ones(fs.shape[0])*2.4), axis=0)
                colorgroup = np.concatenate(
                    (colorgroup, np.ones(ft.shape[0])*2.6), axis=0)
                colorgroup = np.concatenate(
                    (colorgroup, np.ones(x_traj.shape[0])*1.7), axis=0)
                frame_counter = frame_counter+1
        df = pd.DataFrame(dict(xp=xyz[:, 0], yp=xyz[:, 1], zp=xyz[:, 2],
                               framenum=framenum, size=1, colorgroup=colorgroup))
        # plot animation
        fig = px.scatter_3d(df, x="xp", y="yp", z="zp", opacity=alpha,
                            animation_frame="framenum",  # symbol = "framenum",
                            size="size", size_max=10,
                            color="colorgroup", color_continuous_scale=plotly.express.colors.sequential.Turbo, range_color=[0, 4])
        # remove marker outlines from animation
        for i in range(len(fig.frames)):
            fig.frames[i].data[0]['marker']['line'] = dict(
                width=0, color='DarkSlateGrey')
        # plot background
        for i in range(len(keytraces)):
            fig.add_trace(keytraces[i])
        fig.add_trace(BB_trace)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene_dragmode='orbit', scene=dict(
            aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
        plotly.offline.plot(fig, filename=savedir+'fb_'+savename+'.html')

        # save points as nframes, npoints, dim
        np.save(savedir+'x_trajs_'+savename+'.npy', x_trajs.permute((2,0,1)).detach().numpy())
        
        return x_trajs


class MiscTransforms():
    def z_t_to_zt(z, t):
        """
        z: N d
        t: T
        zz: (TN) d
        tt: (TN) 1
        zt: (TN) (d+1)
        """
        zz = torch.tile(z, (t.shape[0], 1))
        tt = t.repeat_interleave(z.shape[0]).reshape((-1, 1))
        zt = torch.cat((zz, tt), dim=1)
        return zt

    def OT_registration(source, target):
        # SCALING EFFECTS IF A PERMUTATION IS RECOVERED OR NOT
        Loss = SamplesLoss("sinkhorn", p=2, blur=0.001, scaling=0.99)
        x = source
        y = target
        a = source[:, 0]*0.0 + 1.0/source.shape[0]
        b = target[:, 0]*0.0 + 1.0/target.shape[0]

        x.requires_grad = True
        z = x.clone()  # Moving point cloud

        # pdb.set_trace()
        if use_cuda:
            torch.cuda.synchronize()

        nits = 5
        for it in range(nits):
            wasserstein_zy = Loss(a, z, b, y)
            # wasserstein_zy = Loss(z, y)
            [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
            z -= grad_z / a[:, None]  # Apply the regularized Brenier map

        if (z.abs() > 10).any().item():
            # ot registration is unstable and overshot.
            dic = {"source": source, "target": target}
            torch.save(dic, "otdebug.tar")
            print("SAVED OT REGISTRATION ERROR")
        return z  # , grad_z

    # return point cloud minimizing W(src,X)*(1-tw) + W(trg,X)*tw
    def unbalanced_OT_Barycenter(src, trg, tw, reach, init=None, tag=""):
            
        src=src.clone().detach()
        trg=trg.clone().detach()
        tw=tw.clone().detach()
        src.requires_grad = False
        trg.requires_grad = False
        
        if tw==0:
            return src
        elif tw==1:
            return trg
        
        if init is None:
            if tw < .5:
                init = src
            else:
                init = trg
                # init = src
                # assymmetry in interpolation possibly due to fixed step size. can mitigate by using consistent initialization.
        
        BC = init.clone().detach()
        BC.requires_grad = True
        
        Loss = SamplesLoss("sinkhorn", p=2, blur=0.0001, scaling = .8, reach=reach)
        nits = 300;
        losses = [];        
        dt = 500;
        for i in range(nits):
            if i==250:
                dt/=2
            
            l1 = Loss(src, BC);
            l2 = Loss(BC, trg);
            loss = l1*(1-tw) + l2*tw;
            [grad_BC] = torch.autograd.grad(loss, [BC])
            with torch.no_grad():
                BC -= grad_BC*dt
                BC.grad = None
            losses.append(loss.item())
            # np.save(tag+'_BC_'+str(i)+'.npy', BC.detach().cpu().numpy())
        
        plt.figure()
        plt.plot(losses, 'k')
        plt.savefig(tag+"_fig.png")
        plt.clf()
        
        # np.save(tag+'_src.npy', src.detach().cpu().numpy())
        # np.save(tag+'_trg.npy', trg.detach().cpu().numpy())
        # np.save(tag+'_tw.npy', tw.detach().cpu().numpy())
        # np.save(tag+'_BC.npy', BC.detach().cpu().numpy())
        # np.save(tag+'_reach.npy', np.array(reach))
        return BC
    
    # works for 2d and 3d.
    def OT_registration_POT_2D(source, target):
        M = ot.dist(source, target)
        M /= M.max()
        n = source.shape[0]
        a, b = torch.ones((n,)) / n, torch.ones((n,)) / n
        Wd = ot.emd(a, b, M)
        _vals, indices = Wd.transpose(0, 1).max(dim=0)
        return target[indices, :], indices

    # MiscTransforms.fill_scene_caches("scenes", n_inner = 10000, n_surface = 10000)

    def fill_scene_caches(scenedir, n_inner=10000, n_surface=10000):
        for scene in glob.glob(scenedir+"/*"):
            for objfile in glob.glob(scene+"/*.obj"):
                print(objfile)
                mesh = MeshDataset(objfile)
                pts_inner, pts_surface = mesh.sample(
                    n_inner=n_inner, n_surface=n_surface)

    # CODE SNIPPET FOR DEBUGGING GEOMLOSS OT_REGISTRATION IF IT BUGS OUT.
    # import Utils; importlib.reload(Utils); from Utils import MiscTransforms
    # dic = torch.load("otdebug.tar");
    # fs = dic["source"]
    # ft = dic["target"]
    # fsp = fs.detach().cpu().numpy()
    # ftp = ft.detach().cpu().numpy()
    # plt.scatter(fsp[:,0], fsp[:,1], s=10, alpha=.5, linewidths=0, c='red', edgecolors='black')
    # plt.scatter(ftp[:,0], ftp[:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
    # z = MiscTransforms.OT_registration(fs,ft)
    # zp = z.detach().cpu().numpy()
    # plt.scatter(zp[:,0], zp[:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')
