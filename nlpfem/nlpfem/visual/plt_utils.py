"""Module to contain plotting utilities"""

import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.patches as patches
import imageio 

def make_gif(gif_path, filenames, duration):
    """Method to make a gif from a set of png files.
    Inputs:
        gif_path: (str) The path to save the gif to 
        filenames: (list) List of file paths to the pngs which we want to make a gif with    
    """
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

def plot_mesh(mesh, time, fig_ax=None, disp_vec=None, show_nn=True, show_dof=True, show_elem_num=True, show_wire=True, 
                show_nodes=True, markersize=5):
    """Mesh to plot the mesh with node numbers and node dof numbers
    Returns:
        fig: (mpl.pyplot)
    """
    if fig_ax is None:
        fig = plt.figure(figsize=[8,4]) 
        ax = fig.add_subplot(111, xlim=(-0.1, mesh.dx+1+0.1), ylim=(-mesh.dy, 2*mesh.dy))
    else:
        ax = fig_ax 
    
    ax.annotate(f'time={time:.5f}', xy=(1, -0.1), xytext=(1, -0.1), fontsize=8)

    if mesh.wall_x is not None:
        ax.plot(np.linspace(mesh.wall_x, mesh.wall_x), np.linspace(-mesh.dy, 2*mesh.dy),'k')

    for i in range(0, mesh.num_nodes):
        # get plotting data 
        Xn_ref = mesh.get_ref_node_position(target=i)
        # add on a displacement if not None 
        if disp_vec is not None:
            u_n = disp_vec[mesh.node_dof_map[i],0]
            x_n = Xn_ref + u_n
        else:
            x_n = Xn_ref

        sx = mesh.elem_side_dx
        sy = mesh.elem_side_dy
        n_dofs = mesh.node_dof_map[i]

        if show_nodes:
            # plot the nodal points in the mesh 
            ax.plot(x_n[0], x_n[1], 'b.', markersize=markersize)

        if show_nn:
            # plot the node numbers
            ax.annotate(f'{i}', xy=(x_n[0], x_n[1]), xytext=(x_n[0]-sx/5, x_n[1]-sy/5), fontsize=8)

        if show_dof:
            # plot the node dof numbers 
            # mark the 1 direction dof
            ax.annotate(f'{n_dofs[0]}', xy=(x_n[0], x_n[1]), xytext=(x_n[0]+sx/6, x_n[1]), fontsize=8)
            # mark the 2 direction dof 
            ax.annotate(f'{n_dofs[1]}', xy=(x_n[0], x_n[1]), xytext=(x_n[0], x_n[1]+sy/6), fontsize=8)

    if show_elem_num:
        for e, element in mesh.element_map.items():
            # get the local node 1 position 
            xn_loc = mesh.ref_node_positions[element.node_nums[0]]

            # plot element numbers
            ec1 = xn_loc[0] + sx/2
            ec2 = xn_loc[1] + sy/2
            # plot element number at center of element
            ax.annotate(f'{e}', xy=(ec1, ec2), xytext=(ec1, ec2), fontsize=8)

            # plot rectangle around element number
            rect = patches.Rectangle((ec1-sx/16, ec2-sy/16), sx/4, sy/4, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    if show_wire:
        for e, element in mesh.element_map.items():
            # get the local node 1 position 
            xy = np.zeros((4,2))
            for i in range(0,4):
                if disp_vec is not None:
                    u_n = disp_vec[mesh.node_dof_map[element.node_nums[i]],0]
                else:
                    u_n = 0
                xn_loc = mesh.ref_node_positions[element.node_nums[i]] + u_n
                xy[i,:] = xn_loc.squeeze()
   
            # plot polygon for the element boundary
            rect = patches.Polygon(xy , linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)

    # plot specifications
    ax.set_aspect('equal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    ax.margins(0.2)      
    plt.tight_layout()

    return ax

def plot_stress(mesh, S_avg_vecs, time, disp_vec, **kwargs):
    """Method to plot the average of stresses on the deformed mesh.
    Inputs:
        mesh: mesh to plot on top of
        S_avg_vecs: (3 x num_elements) array of stress values for each element with 11,22,12 values in each column corresponding to element number
    """
    fig, axes = plt.subplots(3,1, figsize=[15,10])
    uhat = disp_vec

    for k, ax in enumerate(axes):
        ax.set_xlim([-0.5,2.5])
        ax.set_ylim([-0.2,0.4])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        
        minv, maxv = -1e7, 1e7
        if k == 0:
            label='$S_{11}$'
        elif k == 1:
            label='$S_{22}$'
        elif k == 2:
            label='$S_{12}$'
        cmap = plt.cm.inferno
        norm = plt.Normalize(minv, maxv)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.set_title(label)
        plt.colorbar(sm, ax=ax, label=label)

        plot_mesh(mesh=mesh, fig_ax=ax,  time=time, disp_vec=disp_vec, **kwargs)
        for e, element in mesh.element_map.items():
            S_vec_e = S_avg_vecs[k,e]
            
            # get the local node 1 position 
            xy = np.zeros((4,2))
            for i in range(0,4):
                if disp_vec is not None:
                    u_n = disp_vec[mesh.node_dof_map[element.node_nums[i]],0]
                else:
                    u_n = 0
                xn_loc = mesh.ref_node_positions[element.node_nums[i]] + u_n
                xy[i,:] = xn_loc.squeeze()
   
            # plot polygon for the element boundary
            rect = patches.Polygon(xy , linewidth=1, edgecolor='k', facecolor=cmap(norm(S_vec_e)))
            ax.add_patch(rect)


def plot_strain(mesh, E_avg_vecs, time, disp_vec, **kwargs):
    """Method to plot the average of strain on the deformed mesh.
    Inputs:
        mesh: mesh to plot on top of
        E_avg_vecs: (3 x num_elements) array of stress values for each element with 11,22,12 values in each column corresponding to element number
    """
    fig, axes = plt.subplots(3,1, figsize=[15,10])

    for k, ax in enumerate(axes):
        ax.set_xlim([-0.5,2.5])
        ax.set_ylim([-0.2,0.4])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        
        minv, maxv = -0.2, 0.2
        if k == 0:
            label='$E_{11}$'
        elif k == 1:
            label='$E_{22}$'
        elif k == 2:
            label='$E_{12}$'
        cmap = plt.cm.plasma
        norm = plt.Normalize(minv, maxv)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.set_title(label)
        plt.colorbar(sm, ax=ax, label=label)

        plot_mesh(mesh=mesh, fig_ax=ax,  time=time, disp_vec=disp_vec, **kwargs)
        for e, element in mesh.element_map.items():
            E_vec_e = E_avg_vecs[k,e]
            
            # get the local node 1 position 
            xy = np.zeros((4,2))
            for i in range(0,4):
                if disp_vec is not None:
                    u_n = disp_vec[mesh.node_dof_map[element.node_nums[i]],0]
                else:
                    u_n = 0
                xn_loc = mesh.ref_node_positions[element.node_nums[i]] + u_n
                xy[i,:] = xn_loc.squeeze()
   
            # plot polygon for the element boundary
            rect = patches.Polygon(xy , linewidth=1, edgecolor='k', facecolor=cmap(norm(E_vec_e)))
            ax.add_patch(rect)