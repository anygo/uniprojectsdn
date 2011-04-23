package visual;

import framed.*;

import javax.swing.Box;
import javax.swing.JFrame;
import javax.swing.JPanel;
import sampled.*;

import java.awt.BorderLayout;

public class DemoSpectrogramStream {
	
	
	public static void main(String args[]) throws Exception {
		
		JFrame win = new JFrame("Spectrogram");
		win.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		Box box = Box.createVerticalBox();
		win.setContentPane(box);
		
		
		//AudioFileReader reader = new AudioFileReader("/home/cip/2007/sidoneum/praktikum/volume.wav", true);
		//AudioFileReader reader = new AudioFileReader("/home/cip/2007/sidoneum/eclipse_ws/volume.wav", true);
		AudioCapture capt = new AudioCapture("Mikrofon (SoundMAX Integrated D");
		//Synthesizer syn = new ConstantGenerator(220.);
		//syn.setSleepTime(20);
		
		//BufferedAudioSource bas = new BufferedAudioSource(reader);
		
		AudioSource as = capt;
		
		System.err.println(as);
		Window w = new HannWindow(as, 25, 5);
		System.err.println(w);

		
		FrameSource powerspec = new FFT(w);
		System.err.println(powerspec);
		
		
		VisualSpectrogramAudioStream vis = new VisualSpectrogramAudioStream(powerspec, as.getSampleRate(), w.getShift(), 512);
		//vis.setMinMax(0., 100.);
			
		
		//constr.gridx = 0; constr.gridy = 0;
		JPanel panel = new JPanel();
		panel.setLayout(new BorderLayout());
		panel.add(vis, BorderLayout.CENTER);
		box.add(panel);
		win.pack();
		win.setVisible(true);
		
		double [] buf = new double[vis.getFrameSize()];
		
		System.err.println("System running. Close window to terminate application.");
		

		while (vis.read(buf) && win.isVisible()) {

			Thread.sleep(7);

		}
		
	}

}
