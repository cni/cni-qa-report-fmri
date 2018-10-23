[![Docker Pulls](https://img.shields.io/docker/pulls/stanfordcni/cni-qa-report-fmri.svg)](https://hub.docker.com/r/stanfordcni/cni-qa-report-fmri/)
[![Docker Stars](https://img.shields.io/docker/stars/stanfordcni/cni-qa-report-fmri.svg)](https://hub.docker.com/r/stanfordcni/cni-qa-report-fmri/)

# stanfordcni/cni-qa-report-fmri
Calculate QA metrics (displacement, signal spikes, etc.) to create a quality assurance report (png) for an fMRI NIfTI dataset using [code](qa-report-fmri.py) adapted from [CNI/NIMS code](https://github.com/cni/nims/blob/master/nimsproc/qa_report.py) from @rfdougherty.


## What's in the QA report
The QA code is still under active development, so we will likely be adding more metrics to the report in the near future. Right now, the report will give you the QA version number (the version of the QA script that was used to generate the report), as well as the following information:

#### Temporal SNR
This is the median tSNR of all brain voxels (as defined by the [median_otsu algorithm](http://nipy.org/dipy/examples_built/brain_extraction_dwi.html)). The tSNR is computed after the data have been motion-corrected (details below) and the slow-drift has been removed using a 3rd-order polynomial. Note that tSNR is very sensitive to voxel size, with bigger voxels generally producing higher tSNR. The acceleration factor will also affect tSNR (both inplane acceleration and slice multiplex acceleration), with more acceleration leading to lower tSNR. It is also somewhat sensitive to the TR (with longer TRs producing slightly higher tSNR). So it's a useful metric for comparison across scans with the same voxel size, TR and acceleration, but not useful for comparing across scans with different parameters.

#### Spike count
The number of "spikes" detected. The spikes are detected by a simple threshold of the time series z-score plot (details below). The threshold is currently set to 6. This is somewhat arbitrary (and thus why we show the full plots), but we've found that spikes of this magnitude are generally indicative of a problem, such as excessive subject motion or scan hardware issues.

#### Subject Motion
Subject motion is estimated using [FmriRealign4d](http://nipy.org/nipy/stable/api/generated/nipy.algorithms.registration.groupwise_registration.html)  (without slice-time correction). A plot of the mean displacement, both absolute and relative, is presented. Absolute displacement is the mean displacement relative to the middle frame. Relative displacement is the mean displacement relative to the previous frame. (Mean displacement is computed using [Mark Jenkinson's algorithm](http://www.fmrib.ox.ac.uk/analysis/techrep/tr99mj1/tr99mj1/index.html).)

#### Timeseries z-score
A plot of the mean signal (in z-score units) from each slice of the brain. This last plot is useful for detecting spikes in your data, and for determining if the spikes are caused by your subject (e.g., motion) or by a possible problem with the scanner (e.g., white-pixel noise). When a subject moves, even a little, you will often see spikes that span several or all slices. But a white-pixel noise problem typically only affects one slice at a time. Note that the first few time points are ignored for the spike plot.

For this plot (as well as the motion plot) you can get the exact value of any datapoint by hovering your mouse over one of the curves. Also note that the frame numbers start at zero rather than one. Some examples of QA reports are shown below.


## Artifacts that you may find

#### Subject Motion
This is by far the dominant cause of spike-like artifacts in most datasets. Even a small relative head displacement can lead to a signal drop and/or increase. Motion usually affects many slices.

#### White-pixel noise
Spike noise is a common and insidious problem with MR, often caused by a loose screw on the scanner or some small stay piece of metal in the scan room that accumulates energy and then discharges randomly, creating broad-band RF noise at some point during the signal read-out. When this happens, one spot in k-space will have an abnormally  high intensity and show up as a "white pixel". In the image domain, it will often manifest as an abrupt signal drop in one slice at one time-point (a 'spike' in the time series). The problem is particularly acute for EPI scans because of all the gradient blipping during the read-out.

If you see a lot of spike-noise in your data (either motion-induced or from a white-pixel noise problem), there are various tools available to specifically clean up spike-noise artifacts (like AFNI's 3dDespike). FSL's Melodic can also be used to remove artifacts in general (see [fsl_regfit](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC#fsl_regfilt_command-line_program)). You can also try adding the spikes to your GLM as nuisance regressors. If you see a couple of spikes here and there, you might be able to safely ignore them, as they will not have a big effect on most GLM-type analyses. But even one or two spikes can affect certain kinds of correlation analyses, so for that you will have to be more careful.

## Examples of QA reports

__1. A good subject and well-behaved scanner (good subject, good scanner)__

![QA_Good](https://cni.stanford.edu/cniwiki/images/5/56/Qa_good.png)

__2. Subject motion, no scanner spikes (bad subject, good scanner)__

![QA_Motion](https://cni.stanford.edu/cniwiki/images/2/23/Qa_motion.png)

__3. Scanner spikes, little subject motion (good subject, spikey scanner)__

![QA_Spikes](https://cni.stanford.edu/cniwiki/images/0/05/Qa_spikes.png)

## Source Code
The original QA report generation code, from which [this code](qa-report-fmri.py) is adapted, is part of the CNI's NIMS codebase and is [available on Github](https://github.com/cni/nims/blob/master/nimsproc/qa_report.py).
