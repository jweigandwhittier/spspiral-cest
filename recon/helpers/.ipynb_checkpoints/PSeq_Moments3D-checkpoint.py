import numpy as np
import pypulseq as pp

from .PSeq_Base import PSeq_Base


def get_max_difference(start, end, start_pol, end_pol):
    """
    Get the maximum difference (end-start).

    start and end may be None, a scalar, or a list/array, and the maximum difference of
    all combinations is returned.  If either argument is None, will return 0
    """
    if start is None or end is None:
        return 0

    if np.isscalar(start):
        start = [start]
    if np.isscalar(end):
        end = [end]
    if np.isscalar(start_pol):
        start_pol = [start_pol]
    if np.isscalar(end_pol):
        end_pol = [end_pol]

    max_diff = 0
    for val_start_pol in start_pol:
        for val_end_pol in end_pol:
            for val_end in end:
                for val_start in start:
                    diff = val_end_pol * val_end - val_start_pol * val_start
                    max_diff = max(abs(diff), max_diff)

    return max_diff


class PSeq_Moments3D(PSeq_Base):
    """
    Sequence component that plays traps on each axis (or fewer).

    This is made to be fairly flexible in designing the shortest possible time gradients,
    to capture a given starting moment or ending moment.  The intended use is at the end
    of a TR for refocusing/spoiling, but it can also be used for phase encoding or other uses.
    """

    def __init__(
        self,
        pparams,
        start_areas=(None, None, None),
        end_areas=(None, None, None),
        start_polarities=(1, 1, 1),
        end_polarities=(1, 1, 1),
        *args,
        **kwargs,
    ):
        """
        Construct.

        Parameters
        ----------
        start_areas : length 3 list-like
            Areas [1/m] before the start of this component for each gradient channel.  Can be
            None, a single float, or a list-like of floats.  If None, it assumes there
            will be no gradient played on that axis.
            By default (None, None, None)
        end_areas : length 3 list-like
            Areas [1/m] desired at the end of this component for each gradient channel.  Uses
            the same format as start_areas.
            By default (None, None, None)
        start_polarities : length 3 list-like
            Lists possible polarities for each start areas.  Mostly for GIRF measurements,
            otherwise this is just 1
        end_polarities : length 3 list-like
            Lists possible polarities for each end areas.  Mostly for GIRF measurements, 
            otherwise this is just 1

        Notes
        -----
        If the areas provided for starting or ending are a list, the calls to run this
        component will be indexed, using the given list of areas.  Currently this class is
        planned to only handle a single list for each direction, but this can be expanded
        later if needed (TODO).

        All trapezoids are created with the same duration, matching the minimum duration
        needed for largest required area.  For phase-encoding lists of areas, the
        amplitude only is scaled, timing remains the same (TODO: Add options for shortest
        durations on each channel?)
        """
        super().__init__(pparams)

        self.start_areas = start_areas
        self.end_areas = end_areas

        self.start_polarities = start_polarities
        self.end_polarities = end_polarities

        # Calculate the maximum needed area for each channel
        self.max_areas = np.zeros(3)
        for i in range(3):
            self.max_areas[i] = get_max_difference(
                self.start_areas[i],
                self.end_areas[i],
                self.start_polarities[i],
                self.end_polarities[i],
            )

        if not np.any(self.max_areas):
            print('WARNING: PSeq_Moments3D found no areas needed in any direction, will play .1ms delay instead')

        self.active_grad = self.max_areas > 0

        # Get minimum time gradient for each channel, to get final duration
        temp_grads = []
        for i in range(3):
            if self.active_grad[i]:
                _grad = pp.make_trapezoid(
                    channel=self.pparams.channels[i],
                    area=self.max_areas[i],
                    system=self.pparams.system,
                )
                temp_grads.append(_grad)

        # This essentially gets the longest duration from the temp trapezoids
        self.spoil_duration = pp.calc_duration(*temp_grads)

        # Now build up the "final" gradients for each channel, with matched duration
        self.all_grads = []
        self.all_amps = []
        for i in range(3):
            if self.active_grad[i]:
                _grad = pp.make_trapezoid(
                    channel=self.pparams.channels[i],
                    duration=self.spoil_duration,
                    area=self.max_areas[i],
                    system=self.pparams.system,
                )
                self.all_grads.append(_grad)
                self.all_amps.append(_grad.amplitude)
            else:
                self.all_grads.append(None)
                self.all_amps.append(None)

        # Pre-determine which channels have lists of areas, and which it is
        self.all_indexed = []
        for i in range(3):
            if self.active_grad[i]:
                # Throw error if there are multiple areas for both start and end
                if np.ndim(self.start_areas[i]) != 0 and np.ndim(self.end_areas[i]) != 0:
                    err_msg = 'ERROR: PSeq_Moments3D does not currently handle lists for both start and end areas'  # noqa: E501
                    raise Exception(err_msg)

                if np.isscalar(self.start_areas[i]) and np.isscalar(self.end_areas[i]):
                    self.all_indexed.append('scalar')  # Both areas are scalars
                elif np.ndim(self.start_areas[i]) != 0:
                    self.all_indexed.append('start')  # Multiple start areas
                elif np.ndim(self.end_areas[i]) != 0:
                    self.all_indexed.append('end')  # Multiple end areas
            else:
                self.all_indexed.append(None)

    def build_blocks(
        self,
        idx0=0,
        idx1=0,
        idx2=0,
        start_pol0=1.0,
        start_pol1=1.0,
        start_pol2=1.0,
        end_pol0=1.0,
        end_pol1=1.0,
        end_pol2=1.0,
    ):
        """Build all blocks to add to sequence.

        Parameters
        ----------
        idx0, idx1, idx2 : int
            Index for the 0th channel area.  Will be ignored if the areas given for this
            channel were not inputted with multiple values.
        polarities : floats
            Polarities for start and end areas for each channel.  This has been added for
            GIRF measurements, allows the areas to be reversed.

        Returns
        -------
        list of lists
            All blocks to add, outer list is for sequential blocks, inner blocks are all
            components to play within the block.
        """
        all_idx = [idx0, idx1, idx2]
        all_start_pol = [start_pol0, start_pol1, start_pol2]
        all_end_pol = [end_pol0, end_pol1, end_pol2]

        grads_to_play = []
        for i in range(3):
            if self.active_grad[i]:
                # Scale amplitudes to get desired area for a given index
                if self.all_indexed[i] == 'start':
                    start_area = all_start_pol[i] * self.start_areas[i][all_idx[i]]
                else:
                    start_area = all_start_pol[i] * self.start_areas[i]

                if self.all_indexed[i] == 'end':
                    end_area = all_end_pol[i] * self.end_areas[i][all_idx[i]]
                else:
                    end_area = all_end_pol[i] * self.end_areas[i]

                self.all_grads[i].amplitude = (
                    self.all_amps[i] * (end_area - start_area) / self.max_areas[i]
                )
                self.all_grads[i].channel = self.pparams.channels[i]
                
                grads_to_play.append(self.all_grads[i])

        # This is a (time-wasting) hack when no gradient is needed, TODO: handle better.
        if len(grads_to_play) == 0:
            grads_to_play.append(pp.make_delay(.1e-3))
            
        return [grads_to_play]
