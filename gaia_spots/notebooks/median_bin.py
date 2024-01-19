import numpy as np
import astropy 
import matplotlib.pyplot as plt

def median_bin(wave, flux, uncertainty, gap_definition=0.0000001, visualize=False):
    '''
    Group data into chunks, and median-bin them within those chunks.
    
    Parameters 
    ----------
    wave : array 
        The original wavelengths, in microns. 
    flux : array 
        The original fluxes.   
    uncertainty : array 
        The original uncertainties. 
    gap_definition : float 
        If wavelegnths are separated by more than this many micron,
        consider them separate clumps.
    visualize : bool 
        Should we make some visualizations showing details of 
        the binning process or not? 
    
    Returns 
    -------
    binned_wime : array 
        The binned wavelengths.
    binned_flux : array 
        The binned fluxes. 
    binned_uncertainty : array 
        The binned_fluxes.
    '''
    
    '''
    - Sort the data in wave order. '''
    indices = wave.argsort(axis=None)
    _f = flux[indices]
    _e = uncertainty[indices]
    _w = wave[indices] # _w is now a sorted array of wavelengths
    
    '''
    - Identify gap_start as the array indices where np.diff(wave) > gap_definition.'''
    gap_start = np.diff(_w) >= gap_definition # this is a list of T/F values

    '''
    - Create an array of clump start times and clump end times. '''
    edges = _w[1:]*gap_start # This replaces all but the 'border' values with 0
    condition = np.where(edges == 0.) # Array of indices
    compressed = np.delete(edges,condition) - 0.5*gap_definition # Removing the values which are not bin edges (which were 0) and offsetting the edge from the points
    _edges = np.insert(compressed,0,(_w[0] - 0.5*gap_definition)) # create a first edge
    bin_edges = np.append(_edges,np.max(_w) + 0.5*gap_definition) # create a last edge
    if visualize:
        plt.figure(figsize=(5,4))
        plt.title('median binning results')
        # plt.xlim(2459455.65,2459455.7)
        plt.errorbar(_w, _f, _e, label='sorted & unbinned', 
                 marker='.', linewidth=0, elinewidth=1, color='gray')
        for i in range(len(bin_edges)):
            plt.axvline(bin_edges[i],color='k')
            if i == 0:
                plt.axvline(bin_edges[i],color='k',label='bin edges')
    
    '''    
    - Create empty arrays (binned_wave, binned_flux, binned_uncertainty) with one element per clump.'''
    binned_wave = np.array([None]*(len(bin_edges)-1)) # the number of binned times is 1 more than the number of bin_edges
    binned_flux = np.array([None]*(len(bin_edges)-1)) # unless bin_edges includes upper and lower bounds
    binned_uncertainty = np.array([None]*(len(bin_edges)-1))

    # Loop through clumps, select the data points in each clump.
    for i in range(len(bin_edges)-1): 
        start_wave = bin_edges[i]
        end_wave = bin_edges[i+1]
        def_low = np.array(_w > start_wave)
        def_high = np.array(_w < end_wave)
        wrange = (def_low * def_high)
        
        these_waves = _w*wrange
        condition = (these_waves == 0)
        this_clumps_waves = np.delete(these_waves, condition)
        these_fluxes = _f*wrange
        condition = (these_fluxes == 0)
        this_clumps_fluxes = np.delete(these_fluxes, condition)
        these_errs = _e*wrange
        condition = (these_errs == 0)
        this_clumps_uncertainties = np.delete(these_errs, condition)
        
        binned_wave[i] = np.median(this_clumps_waves) # Set the binned_wave for this clump to be the median of the wavelengths in the clump.
        binned_flux[i] = np.median(this_clumps_fluxes) # Set the binned_flux for this clump to be median of the fluxes in the clump.
        binned_uncertainty[i] = astropy.stats.mad_std(this_clumps_fluxes)/np.sqrt(len(this_clumps_waves)) #  `astropy.stats.mad_std` to be less sensitive than `np.std` to outliers.
        #  # To avoid problems with small numbers of points in a bin, don't let uncertainty be smaller than...
        if binned_uncertainty[i] <= np.median(this_clumps_uncertainties)/np.sqrt(len(this_clumps_waves)) :
            binned_uncertainty[i] = np.median(this_clumps_uncertainties)/np.sqrt(len(this_clumps_waves))
        # if binned_uncertainty[i] <= 0.01:
        #     binned_uncertainty[i] = 0.01
    
    if visualize:
        plt.errorbar(binned_wave, binned_flux, binned_uncertainty,
                     label='binned', linewidth=0,elinewidth=1, color='red',zorder=20)
        plt.legend()
    
    return binned_wave, binned_flux, binned_uncertainty