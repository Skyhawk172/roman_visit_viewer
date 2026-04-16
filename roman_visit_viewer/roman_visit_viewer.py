#!/usr/bin/env python
# coding: utf-8

# This code converts a visit file into Exposure objects one block at a time.

# VisitFileParser
#     ↓ (iterator)
# ExposureIterator  → yields raw text blocks
#     ↓
# ExposureParser    → converts block → Exposure object
#     ↓
# Exposure (data + behavior)
#     ↓
# Plotting layer

import argparse

from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt 

import numpy as np

from pathlib import Path

import pysiaf

from dataclasses import dataclass, field
from typing import Optional

import re, os


# # Functions and Classes

RSIAF = pysiaf.Siaf("Roman")

def add_compass_lower_right(ax, wcs, size=8*u.arcmin, pad=0.05,
                           color="white"):
    """
    Add N/E compass fixed to lower-right corner of a WCSAxes.
    """

    # Transform from axis values to pixel data: 
    x_ax = 1 - pad
    y_ax = pad


    x_pix, y_pix = ax.transAxes.transform((x_ax, y_ax))
    x_pix, y_pix = ax.transData.inverted().transform((x_pix, y_pix))
    
    # Pixel → sky coordinate
    center = SkyCoord.from_pixel(x_pix, y_pix, wcs)

    # Compute N and E directions
    north = SkyCoord(center.ra, center.dec + size)

    east = SkyCoord( center.ra + size / np.cos(center.dec), center.dec)

    # Draw arrows
    trans = ax.get_transform("icrs")

    arrow_props = dict(arrowstyle='-|>', color=color, lw=1, shrinkA=0, shrinkB=0, alpha=0.5)
    for tip in [north, east]:
        ax.annotate("",
            xy=(tip.ra.deg, tip.dec.deg),
            xytext=(center.ra.deg, center.dec.deg),
            arrowprops=arrow_props,
            xycoords=trans,
            textcoords=trans)
    

    # Add labels
    ax.text(north.ra.deg, north.dec.deg, "N", ha='center', va='bottom', alpha=0.5,
            color=color, transform=trans)

    ax.text(east.ra.deg, east.dec.deg, "E", ha='right', va='center', alpha=0.5,
            color=color, transform=trans)

def roman_attitude(q):
    '''
    Calculate the RA, Dec, and V3PA based on input quaternion from visit file.
    Quaternion rotates ECI → BCS (scalar-last convention).

    Parameters
    ----------
    q : list
        Quaternion

    Returns
    ---------
    ra, dec, pa_v3
    '''

    x,y,z,w = q  # scalar-last

    # rotation matrix (ECI→BCS)
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])

    V1 = R[:,0]      # pointing
    V3 = R[:,2]      # +V3 (Roman definition)

    # --- RA/DEC ---
    V1 /= np.linalg.norm(V1)

    dec = np.arcsin(V1[2])
    ra  = np.arctan2(V1[1], V1[0])
    if ra < 0:
        ra += 2*np.pi

    # --- PA(+V3) ---
    Z = np.array([0.,0.,1.])

    N = Z - np.dot(Z,V1)*V1
    N /= np.linalg.norm(N)

    E = np.cross(N, V1)

    V3 -= np.dot(V3,V1)*V1
    V3 /= np.linalg.norm(V3)

    pa_v3 = np.degrees(np.arctan2(
        np.dot(V3,E),
        np.dot(V3,N)
    )) % 360

    return np.degrees(ra), np.degrees(dec), pa_v3



def retrieve_2mass_image(ra, dec, visitname, verbose=True, redownload=False, filter='J', fov=1.8):
    """Obtain from Aladin a 2MASS image for the pointing location of a JWST visit

    Uses HIPS2FITS service; see http://alasky.u-strasbg.fr/hips-image-services/hips2fits

    FITS files for the retrieved images are cached for re-use, in a subdirectory
    `image_cache` next to where this code is.

    Parameters
    ----------
    visit : VisitFileContents object
        Representation of some JWST visit file
    filter : string
        Which bandpass filter in 2MASS to get?
    verbose : bool
        more output text
    redownload : bool
        Even if image is already downloaded and cached, ignore that and download from Vizier again.

    """

    hips_catalog = f'CDS/P/2MASS/{filter}'  # also try 2MASS/color
    width = 1024
    height = 1024

    if fov!= 0.35:
        # if a non-default FOV is used, save that specially
        img_fn = os.path.join(f'img_2mass_{filter}_{visitname.strip(".vst")}_fov{fov}.fits')
    else:
        img_fn = os.path.join(f'img_2mass_{filter}_{visitname.strip(".vst")}.fits')

    if not os.path.exists(img_fn) or redownload:

        # optional / TBD - add PA into this query?
        # rotation_angle=90.0
        url = f'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?hips={(hips_catalog)}&width={width}&height={height}&fov={fov}&projection=TAN&coordsys=icrs&ra={ra}&dec={dec}'

        if verbose:
            print(f"Retrieving 2MASS image from Aladin near ra={ra} & dec={dec}...")

        with fits.open(url) as hdu:
            hdu.writeto(img_fn, overwrite=True)
            if verbose:
                print(f"   Saved to {img_fn}")
    hdu = fits.open(img_fn)
    return hdu



def plot_all_exposures(parser, exp_num, image_hdu, wcs, fig=None, ax=None, **kwargs):
    ax.coords[1].set_ticklabel_visible(False)  
    
    exposures = list(parser)
    ndithers = len(exposures)

    norm = ImageNormalize(image_hdu[0].data, interval=PercentileInterval(99.99), stretch=AsinhStretch(a=0.0001))    
    ax.imshow(image_hdu[0].data, cmap='magma', norm=norm, origin='lower', zorder=-50)  # negative zorder to be below pysiaf aperture fill zorder   

    add_compass_lower_right(ax, wcs)

    overlay = ax.get_coords_overlay('icrs')
    overlay.grid(color='white', ls='dotted', alpha=0.5)
    
    for i, iexp in enumerate(exposures):
        if i == (exp_num-1):
            color='yellow'
            zorder = 99
        else:
            color='cyan'
            zorder=1
        
        ra_v1, dec_v1, v3pa_v1 = iexp.radec

        att_mat = pysiaf.rotations.attitude_matrix(0, 0, ra_v1, dec_v1, v3pa_v1)
        boresight = RSIAF['BORESIGHT']
        boresight.set_attitude_matrix(att_mat)
        
        for isca in range(1, 19):
            aper = RSIAF[f'WFI{isca:02d}_FULL']
            aper.set_attitude_matrix(att_mat)
            aper.plot(frame='sky', fill_alpha = 0.15, transform=ax.get_transform('icrs'), ax=ax, color=color, **kwargs, zorder=zorder)

    ax.text(0.025, 0.975, f"{parser.visit_name.strip(".vst"):}", color='white', transform=ax.transAxes,
           fontsize=12, verticalalignment='top')
 
    if ndithers > 1:
       ax.text(0.025, 0.94, f"{ndithers:} Dithers", color='white', transform=ax.transAxes,
           fontsize=12, verticalalignment='top') 

    ax.set_xlabel("ra")
    ax.set_ylabel("dec")

    return 



def plot_manager(parser, exp_num=1, savefig=True):
    '''
    Plot content parsed from visit file.
    If multiple exposures (a.k.a. dithers), will plot two subplots, else just one.

    Parameters:
    -----------
    Parser: VisitFileParser Object
    
    exp: int (optional)
        exposure to plot 

    Return:
    -----------
    None
    
    '''
    all_exps = list(parser)
    ndithers = len( all_exps )

    if exp_num > ndithers:
        print(f"Requested exposure {exp_num:} but visit file contains only {ndithers:} exposures; defaulting to first exposure")
        exp_num = 1

    exposure = all_exps[exp_num-1]
    
    ra_v1, dec_v1, v3pa_v1 = exposure.radec
    
    att_mat = pysiaf.rotations.attitude_matrix(0, 0, ra_v1, dec_v1, v3pa_v1)
    boresight = RSIAF['BORESIGHT']
    boresight.set_attitude_matrix(att_mat)
    wfi_cen = RSIAF['WFI_CEN']
    ra_wfi, dec_wfi = pysiaf.rotations.tel_to_sky(att_mat, wfi_cen.V2Ref, wfi_cen.V3Ref)
    ra_wfi = ra_wfi.to(u.deg).value
    dec_wfi= dec_wfi.to(u.deg).value
    
    image_hdu = retrieve_2mass_image(ra_wfi, dec_wfi, exposure.visit_name, redownload=False)
    wcs = WCS(image_hdu[0].header)
    
    if ndithers > 1:
        fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': wcs}, figsize=(20,9))
        axes = ax.flatten()
    else:
        fig = plt.figure(figsize=(20,9), dpi=100)
        axes = [plt.subplot(projection=wcs)]

    exposure.plot(fig, axes[0], savefig=False, ndithers=ndithers)
        
    if ndithers > 1: 
        plot_all_exposures(parser, exp_num, image_hdu, wcs, fig=None, ax=axes[1])

    if savefig:
        savename = parser.visit_name.replace(".vst", "_all.png")
        fig.savefig(savename)



@dataclass
class Exposure:
    '''
    Object to store the exposure blocks from the visit file
    '''
    visit_name: Optional[str] = None 
    exp_id: Optional[int] = None
    quaternion: Optional[tuple[float, float, float, float]] = None
    guide_mode: Optional[str] = None  
    gs_cmds: list[str] = field(default_factory=list)
    fwa: Optional[str] = None
    matab: Optional[str] = None

    @property
    def radec(self):
        if not hasattr(self, "_radec"):
            self._radec = roman_attitude(self.quaternion)
        return self._radec

    def plot(self, fig=None, ax=None, savefig=True, ndithers=None):
        
        ra_v1, dec_v1, v3pa_v1 = self.radec

        att_mat = pysiaf.rotations.attitude_matrix(0, 0, ra_v1, dec_v1, v3pa_v1)
        boresight = RSIAF['BORESIGHT']
        boresight.set_attitude_matrix(att_mat)
        wfi_cen = RSIAF['WFI_CEN']
        ra_wfi, dec_wfi = pysiaf.rotations.tel_to_sky(att_mat, wfi_cen.V2Ref, wfi_cen.V3Ref)
        ra_wfi = ra_wfi.to(u.deg).value
        dec_wfi= dec_wfi.to(u.deg).value
        v3pa_wfi = pysiaf.rotations.posangle(att_mat, wfi_cen.V2Ref, wfi_cen.V3Ref)

        image_hdu = retrieve_2mass_image(ra_wfi, dec_wfi, self.visit_name, redownload=False)
        wcs = WCS(image_hdu[0].header)

        if not ax:
            fig = plt.figure(figsize=(20,9), dpi=100)
            ax = plt.subplot(projection=wcs)

        overlay = ax.get_coords_overlay('icrs')
        overlay.grid(color='white', ls='dotted', alpha=0.5)

        norm = ImageNormalize(image_hdu[0].data, interval=PercentileInterval(99.99), stretch=AsinhStretch(a=0.0001))    
        ax.imshow(image_hdu[0].data, cmap='magma', norm=norm, origin='lower', zorder=-50)  # negative zorder to be below pysiaf aperture fill zorder    

        
        ax.scatter(ra_wfi, dec_wfi, marker='x', s=30, color='yellow', transform=ax.get_transform('icrs'), zorder=99)
        ax.scatter(ra_v1, dec_v1, marker='x', s=30, color='white', transform=ax.get_transform('icrs'))
        plt.annotate( "V1 axis", xy=(ra_v1, dec_v1), xytext=(-18,7), color='white',
                      xycoords=ax.get_transform('icrs'), textcoords="offset points")
        
        
        n_gs = 0
        for isca in range(1, 19):
            color = 'darkgrey' if isca==16 else 'cyan'
            aper = RSIAF[f'WFI{isca:02d}_FULL']
            aper.set_attitude_matrix(att_mat)
            aper.plot(frame='sky', fill_alpha = 0.15, transform=ax.get_transform('icrs'), ax=ax, color=color)
    
            if self.guide_mode != "COARSE":
                if self.gs_cmds[isca-1][1] != '"SKY_FIXED"':
                    n_gs += 1
                    # The FGS frame is centered on the detector, but Y is going "down"
                    # Convert these to science frame first, then to sky:
                    scix = float(self.gs_cmds[isca-1][2]) + 2048
                    sciy = 2048 - float(self.gs_cmds[isca-1][3])
                    gs_ra, gs_dec = aper.sci_to_sky(scix, sciy)
                    ax.plot(gs_ra, gs_dec, 'o', color='yellow', markersize=8, fillstyle='none', transform=ax.get_transform('icrs'), zorder=99)
    
        if ndithers:
            ax.text(0.025, 0.975, f"{self.visit_name.strip(".vst"):} - exposure {self.exp_id+1:} of {ndithers:}", color='white', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')
        else:
            ax.text(0.025, 0.975, f"{self.visit_name.strip(".vst"):} - exposure {self.exp_id+1:}", color='white', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')

        ax.text(0.025, 0.94, f"RA@WFI_CEN     = {ra_wfi:5.2f}°", color='white', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')
        ax.text(0.025, 0.91, f"Dec@WFI_CEN   = {dec_wfi:5.2f}°", color='white', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')
        ax.text(0.025, 0.88, f"V3PA@WFI_CEN = {v3pa_wfi:5.2f}°", color='white', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')
        
        ax.text(0.025, 0.83, f"{self.guide_mode:} - {n_gs:} guide stars", color='white', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')
    
        ax.text(0.025, 0.78, f"{self.fwa:}   {self.matab:}", color='white', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')
    
        ax.set_xlabel("ra")
        ax.set_ylabel("dec")

        add_compass_lower_right(ax, wcs)

        if savefig:
            savename = self.visit_name.replace(".vst", f"_exp{self.exp_id+1:02d}.png")
            fig.savefig(savename)



class ExposureParser:
    '''
    Class to parse each exposure block from visit file
    '''
    
    QUAT_RE = re.compile(r'SCF_AM_SLEW_MAIN_F\(([^)]+)\)')
    GS_RE   = re.compile(r'WFIF_FGS_GSDS_ENTRY_F\(([^)]+)\)')
    FILTER_RE = re.compile(r'WFI_MCE_EWA_MOVE_ABS_F\(([^)]+)\)')
    MATAB_RE = re.compile(r'WFI_LOAD_SCI_MA_SETREADFRMS_F\(([^)]+)\)')

    def parse(self, lines, exp_id, visit_name):

        block = Exposure() #raw_lines=lines)
        
        block.exp_id = exp_id
        block.visit_name = visit_name
        
        for line in lines:
            match = self.QUAT_RE.search(line)
            if match:
                values = match.group(1).split(",")[:4]
                block.quaternion = tuple(map(float, values))

            match = self.GS_RE.search(line)
            if match:
                pattern = r'\s*(\w+),\s*(\d+),\s*(\w+)\((.*?)\);\s*(.*)'
                m = re.match(pattern, line)
                if m:
                    arg_string  = m.group(4)
                    block.gs_cmds.append(re.findall(r'"[^"]*"|[^,]+', arg_string) )

            # GET FILTER
            match = self.FILTER_RE.search(line)
            if match:
                block.fwa = match.group(1).split("_")[-1].strip('"')
    
            # GET MATAB
            match = self.MATAB_RE.search(line)
            if match:
                block.matab = match.group(1).split(",")[1].strip('"')[4:]

        if len( block.gs_cmds ) > 1:
            block.guide_mode = "TRACK" 
        else:
            block.guide_mode = "COARSE"

        return block
        



class ExposureIterator:
    '''
    Iterator class to iterate through blocks in the visit file
    '''

    # This string is the end of each block in visit file
    END_RE = re.compile( r'ACT,\s*01,\s*SCF_AC_HGA_MODE_F\("TRACK"\);\s*HGA_MODE' )
    
    def __init__(self, filename):
        self.filename = filename
        
    def __iter__(self):
        block = []

        with open(self.filename) as f:

            apt_template = None
            for i, line in enumerate(f):
                if i==0 and line.startswith(";@"):
                    apt_template = line[2:].strip()
                if apt_template != "WFI Fundamental Observation Sequence":
                    raise VisitFileParsingError(f"Visit Template not supported - {apt_template:}")
                    
                line = line.rstrip()
                block.append(line)

                if self.END_RE.search(line):
                    yield block
                    block = []
        if block:
            yield block


class VisitFileParsingError(Exception):
    pass
    
class VisitFileParser:
    '''
    Main class for parsing a visit file.
    '''
    
    def __init__(self, filename):
        self.filename = filename
        self.visit_name = self.filename.split("/")[-1]

        # Iterator through each exposure:
        self.iterator = ExposureIterator(self.filename)

        # Parse each exposure
        self.exposures = ExposureParser()

    def __iter__(self):
        for exp_id, raw_block in enumerate(self.iterator):
            exposure = self.exposures.parse(raw_block, exp_id, self.visit_name)
            yield exposure


def main():

    argparser = argparse.ArgumentParser(description="Parse Roman visit file and plot exposures")
            
    argparser.add_argument("visit_file", help="Full path to .vst visit file")
    argparser.add_argument("--exp_num", type=int, default=1, help="Exposure number to plot (default: 1)")
    argparser.add_argument("--plot_all", action="store_true", help="Plot and save every exposure instead (default: False)")

    args = argparser.parse_args()

    visit_path = Path(args.visit_file)
    if not visit_path.exists():
        raise FileNotFoundError(f"{visit_path} not found")

    
    visit_parser = VisitFileParser(str(visit_path))
    
    plot_all = args.plot_all

    if plot_all:
        for block in visit_parser:
            block.plot( savefig=True )
    else:
        plot_manager( visit_parser, exp_num = args.exp_num, savefig=True )



if __name__ == "__main__":
    main()




