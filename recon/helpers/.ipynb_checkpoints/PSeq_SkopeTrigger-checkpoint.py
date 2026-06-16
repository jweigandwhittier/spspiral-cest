import pypulseq as pp

from .PSeq_Base import PSeq_Base


class PSeq_SkopeTrigger(PSeq_Base):
    """Sequence component that plays a trigger for the field camera."""

    def __init__(self, pparams, duration=200e-6, *args, **kwargs):
        """
        Construct.

        Parameters
        ----------
        pseq : Pseq_Base derived, optional
            Another Base sequence or sequence component to copy parameters from.
            By default None
        duration : float, optional
            Time in [seconds] for this whole block.  The time here is a little bigger than
            the inherent delay of the Skope system (~150-160us).  So this makes sure
            nothing is happening when the system is switching from TX to RX. And any subsequent
            gradients can start immediately if needed.
            By default 200e-6
        """
        super().__init__(pparams)

        self.duration = duration
        self.trig = pp.make_digital_output_pulse(
            channel='ext1',
            duration=self.pparams.system.grad_raster_time,
        )
        self.pp_delay = pp.make_delay(self.duration - self.trig.duration)

    def build_blocks(self):
        """Build all blocks to add to sequence.

        Returns
        -------
        list of lists
            All blocks to add, outer list is for sequential blocks, inner blocks are all
            components to play within the block.
        """
            
        return [[self.trig, self.pp_delay]]

    def get_duration_from_excite(self):
        """
        Get time since excitation of this sequence component.

        TODO
        ----
        Figure out exactly when the Skope triggers relative to this element.  Currently we
        are assuming that it is at the end of the self.trig, but this seems unlikely.
        """
        dur = pp.calc_duration(self.trig, self.pp_delay) - pp.calc_duration(self.trig)
        return dur
