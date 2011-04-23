package visual;

import framed.*;
import javax.swing.JFrame;
import sampled.*;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;

public class VisualMFCC {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {

		//AudioCapture capt = new AudioCapture("Headset [plughw:1,0]");
		//AudioCapture capt = new AudioCapture("Intel [plughw:0,0]");
		
		Synthesizer syn = new SineGenerator(new double [] { 440. });
		syn.setSleepTime(250);
		
		//AudioFileReader reader = new AudioFileReader("/home/cip/2007/sidoneum/praktikum/volume.wav", true);
		
		AudioSource as = syn;
		System.err.println(as);
		
		Window w = new HammingWindow(as, 25, 10);
		System.err.println(w);
		
		Visualizer1D vis0 = new Visualizer1D(w, w.toString(), false);
		vis0.setMinMax(-1., +1.);
		
		FrameSource powerspec = new FFT(vis0);
		System.err.println(powerspec);
		
		Visualizer1D vis1 = new Visualizer1D(powerspec, powerspec.toString(), true);
		vis1.setMinMax(0., 10.);
		
		Visualizer2D vis4 = new Visualizer2D(powerspec, "spectrogram (log DFT)", true);
		vis4.setMinMax(0., 10.);
		
		FrameSource mel = Melfilter.create(vis4, as.getSampleRate(), "-1,-1,-1,24");
		System.err.println(mel);
		
		Visualizer1D vis2 = new Visualizer1D(mel, mel.toString(), false);
		vis2.setMinMax(Math.log(1E-6), 10.);
		
		FrameSource dct = new DCT(vis2, true, true);
		System.err.println(dct);
		
		Visualizer1D vis3 = new Visualizer1D(dct, dct.toString(), false);
		vis3.setMinMax(-10., 10.);
		
		Visualizer2D vis5 = new Visualizer2D(vis3, "mfcc", false);
		vis5.setMinMax(-10., 10.);
		
		double [] buf = new double [dct.getFrameSize()];
		
		JFrame win = new JFrame("mfcc computation");
		
		GridBagLayout layout = new GridBagLayout();
		GridBagConstraints constr = new GridBagConstraints();
		
		win.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		win.getContentPane().setLayout(layout);
		
		constr.gridx = 0; constr.gridy = 0;
		win.getContentPane().add(vis0, constr);
		
		constr.gridx = 0; constr.gridy = 1;
		win.getContentPane().add(vis1, constr);
		
		constr.gridx = 1; constr.gridy = 0;
		win.getContentPane().add(vis2, constr);
		
		constr.gridx = 0; constr.gridy = 2;
		win.getContentPane().add(vis3, constr);
	
		constr.gridx = 1; constr.gridy = 1;
		win.getContentPane().add(vis4, constr);
		
		constr.gridx = 1; constr.gridy = 2;
		win.getContentPane().add(vis5, constr);
		
		win.pack();
		
		win.setVisible(true);
		
		// measure frames per second
		
		System.err.println("system running. close window to terminate application");
		
		long ts = System.currentTimeMillis();
		long frm = 0;
		boolean speech = false;
		double fps = 0;
		while (vis5.read(buf) && win.isVisible()) {
			double elapsed = (System.currentTimeMillis() - ts) / 1000;
			++frm;
			if (elapsed > 1) {
				fps = frm / elapsed;
				frm = 0;
				ts = System.currentTimeMillis();
			}

			if (buf[0] > -10.)
				speech = true;
			else
				speech = false;
			
			win.setTitle("mfcc computation (" + fps + "fps): " + (speech ? "speech..." : "silence"));
		}
	}

}
