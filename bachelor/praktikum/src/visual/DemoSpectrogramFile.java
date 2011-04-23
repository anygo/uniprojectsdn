package visual;

import javax.sound.sampled.UnsupportedAudioFileException;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSlider;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import sampled.*;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.File;
import java.io.IOException;

/**
 * Demo Application to test features of the spectrogram visualizer for audio files
 * 
 * @author sidoneum, sifeluga
 * 
 */

public class DemoSpectrogramFile extends JPanel implements ActionListener, ChangeListener, ItemListener {

	private static final long serialVersionUID = 1L;
	
	private JCheckBox checkBoxColored;
	private JCheckBox checkBoxAveraged;
	private JCheckBox checkBoxLog;
	private JCheckBox checkBoxFrameDiagrams;
	
	private JLabel labelLength;
	private JLabel labelShift;

	private JFrame diagrams;
	private VisualSpectrogramAudioFile vis = null;
	private VisualSpectrumDiagram vs = null;
	private VisualAutocorrelationDiagram va = null;
	private VisualMelDiagram vm = null;
	private VisualMFCCDiagram mfcc = null;
	
	private BufferedAudioSource bas;
	
	private int windowLength = 25;
	private int windowShift = 10;
	private String windowType = "hamm";


	public static void main(String args[]) throws Exception {
		
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
	}
		
	private static void createAndShowGUI() {	

		JFrame win = new JFrame("Spectrogram");
		win.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		JComponent newContentPane = new DemoSpectrogramFile();
		newContentPane.setOpaque(true);
		
		win.setContentPane(newContentPane);
				
		win.pack();
		win.setVisible(true);
		
	}
		
	public DemoSpectrogramFile() {
		super(new BorderLayout());	
		
		final JFileChooser fc = new JFileChooser();
		System.out.println("choose a .wav-file");
		fc.setFileFilter(new FileNameExtensionFilter(".wav-files", "wav"));
		fc.showOpenDialog(this);
		File file = fc.getSelectedFile();

		AudioFileReader reader = null;
		try {
			reader = new AudioFileReader(file.toString(), true);
		} catch (UnsupportedAudioFileException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
	
		AudioSource as = reader;
		
		bas = new BufferedAudioSource(as);
		
		vis = new VisualSpectrogramAudioFile(bas); 
		vis.setDataRange(0, 10000000);
		
		JSlider sliderWindowLength = new JSlider(JSlider.VERTICAL, 	1, 64, 25);
		JSlider sliderWindowShift = new JSlider(JSlider.VERTICAL, 	1, 64, 10);
		sliderWindowLength.addChangeListener(this);
		sliderWindowShift.addChangeListener(this);
		sliderWindowLength.setToolTipText("windowLength");
		sliderWindowShift.setToolTipText("windowShift");
		
		
		
		
		checkBoxAveraged = new JCheckBox("averaged");
		checkBoxAveraged.setSelected(false);
		
		checkBoxLog = new JCheckBox("log scale");
		checkBoxLog.setSelected(false);
		
		checkBoxColored = new JCheckBox("colored");
		checkBoxColored.setSelected(false);
		
		checkBoxFrameDiagrams = new JCheckBox("frame diagrams");
		checkBoxFrameDiagrams.setSelected(false);
		
		checkBoxAveraged.addItemListener(this);
		checkBoxLog.addItemListener(this);
		checkBoxColored.addItemListener(this);
		checkBoxFrameDiagrams.addItemListener(this);
			
		
		JRadioButton buttonHamming = new JRadioButton("Hamming");
		buttonHamming.setActionCommand("Hamming");
		buttonHamming.setSelected(true);
		
		JRadioButton buttonHanning = new JRadioButton("Hanning");
		buttonHanning.setActionCommand("Hanning");
		
		JRadioButton buttonRectangle = new JRadioButton("Rectangle");
		buttonRectangle.setActionCommand("Rectangle");
		
		ButtonGroup windowTypeGroup = new ButtonGroup();
		windowTypeGroup.add(buttonHamming);
		windowTypeGroup.add(buttonHanning);
		windowTypeGroup.add(buttonRectangle);
		
		buttonHamming.addActionListener(this);
		buttonHanning.addActionListener(this);
		buttonRectangle.addActionListener(this);
		
		JPanel radioPanel = new JPanel(new GridLayout(0, 1));
        radioPanel.add(buttonHamming);
        radioPanel.add(buttonHanning);
        radioPanel.add(buttonRectangle);
		
        JSlider sliderBrightness = new JSlider(0, 100, 100 - (int)(vis.getBrightness() * 100));
        sliderBrightness.addChangeListener(this);
        sliderBrightness.setToolTipText("brightness");
        
        JSlider sliderContrast = new JSlider(800, 1000, 1000-(int)(vis.getContrast()));
        sliderContrast.addChangeListener(this);
        sliderContrast.setToolTipText("contrast");
        
        Box sliderContrastBox = Box.createHorizontalBox();
        sliderContrastBox.add(new   JLabel("Contrast     "));
        sliderContrastBox.add(sliderContrast);
        
        Box sliderBrightnessBox = Box.createHorizontalBox();
        sliderBrightnessBox.add(new JLabel("Brightness"));
        sliderBrightnessBox.add(sliderBrightness);
        
        Box adjustors = Box.createVerticalBox();
        adjustors.add(sliderContrastBox);
        adjustors.add(sliderBrightnessBox);
		
		JPanel panel = new JPanel();
		Box configBox = Box.createVerticalBox();
	
		
		labelLength = new JLabel("Length: 25");
		labelShift = new JLabel("Shift: 10");
		
		Box sliders = Box.createHorizontalBox();
		sliders.add(sliderWindowLength);
		sliders.add(sliderWindowShift);
		sliders.add(radioPanel);

		configBox.add(sliders, BorderLayout.EAST);
		
		Box checkBoxes = Box.createVerticalBox();
		checkBoxes.add(checkBoxColored);
		checkBoxes.add(checkBoxAveraged);
		checkBoxes.add(checkBoxLog);
		checkBoxes.add(checkBoxFrameDiagrams);
		
		
		JButton buttonPrint = new JButton();
		buttonPrint.addActionListener(this);
		buttonPrint.setText("Print to File");
		
		JButton buttonShowAll = new JButton();
		buttonShowAll.addActionListener(this);
		buttonShowAll.setText("Show All");
		
		Box windowInfoBox = Box.createVerticalBox();
		windowInfoBox.add(new JLabel("Window Function"));
		windowInfoBox.add(labelLength);
		windowInfoBox.add(labelShift);
		windowInfoBox.setBorder(BorderFactory.createEtchedBorder());
		
		JPanel buttons = new JPanel(new FlowLayout());
		buttons.add(adjustors);
		buttons.add(checkBoxes);
		buttons.add(buttonShowAll);
		buttons.add(buttonPrint);
		buttons.add(new JLabel(" "));
		buttons.add(windowInfoBox);
		

		panel.setLayout(new BorderLayout());
		panel.add(vis, BorderLayout.CENTER);
		panel.add(configBox, BorderLayout.EAST);
		panel.add(buttons, BorderLayout.SOUTH);
		add(panel);

		
		initFrameBasedGraphs(as.getSampleRate());
		
		vis.addFrameListener(vs);
		vis.addFrameListener(va);
		vis.addFrameListener(vm);
		vis.addFrameListener(mfcc);
		
	}
	
	private void initFrameBasedGraphs(int sampleRate) {
	  diagrams = new JFrame("Visualizer"); 
	  Box box = Box.createVerticalBox();
	  diagrams.setContentPane(box);  
	  diagrams.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
	  
	  vs = new VisualSpectrumDiagram("Spektrum", true, sampleRate / 2);
	  vs.setMinMax(0, 100);
	  vs.setWinType("hamm");
	  vs.setWinLen(25);
	    
	  va = new VisualAutocorrelationDiagram("Autokorrelation", sampleRate / 2);
	  va.setMinMax(-1000, 1000);
	  va.setWinType("hamm");
	  va.setWinLen(25);
	  
	  vm = new VisualMelDiagram("Mel Attribute");
	  vm.setMinMax(-10, 10);
	  vm.setWinType("hamm");
	  vm.setWinLen(25);
	  
	  mfcc = new VisualMFCCDiagram("MFCC");
	  mfcc.setMinMax(-80, 80);
	  mfcc.setWinType("hamm");
	  mfcc.setWinLen(25);
	  
	  JPanel panel = new JPanel();
	  panel.setBorder(new EmptyBorder(10, 10, 10, 10));
	  panel.setLayout(new BorderLayout());
	  panel.add(vs, BorderLayout.CENTER);
	  box.add(panel);    
	  JPanel panel4 = new JPanel();
	  panel4.setBorder(new EmptyBorder(10, 10, 10, 10));
	  panel4.setLayout(new BorderLayout());
	  panel4.add(va, BorderLayout.CENTER);
	  box.add(panel4);
	  JPanel panel2 = new JPanel();
	  panel2.setBorder(new EmptyBorder(10, 10, 10, 10));
	  panel2.setLayout(new BorderLayout());
	  panel2.add(vm, BorderLayout.CENTER);
	  box.add(panel2);
	  JPanel panel3 = new JPanel();
	  panel3.setBorder(new EmptyBorder(10, 10, 10, 10));
	  panel3.setLayout(new BorderLayout());
	  panel3.add(mfcc, BorderLayout.CENTER);
	  box.add(panel3);
	      
	  Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
	  int top = (screenSize.height - diagrams.getHeight()) / 2;
	  int left = (screenSize.width - diagrams.getWidth()) / 2;
	  
	  diagrams.setLocation(left, top);
	  diagrams.pack();
	  diagrams.setSize(320, 640);
	  diagrams.setVisible(false);
}
	
	
	@Override
	public void actionPerformed(ActionEvent e) {
		
		if (e.getActionCommand() == "Print to File") {
			vis.writePNGImageToFile();
		}
		else if (e.getActionCommand() == "Show All") {
			vis.setDataRange(0, bas.getBufferSize());
		}
		else if (e.getActionCommand() == "Hamming") {
			windowType = "hamm";
			vis.setWindowConfig(windowType, windowLength, windowShift);
			vis.computeAndRepaintSpectrogram();

			vs.setWinType("hamm");
			vs.setWinLen(windowLength);
			try {
				vs.update();
			} catch (Exception e1) {
				e1.printStackTrace();
			}
		
		}
		else if (e.getActionCommand() == "Hanning") {
			windowType = "hann";
			vis.setWindowConfig(windowType, windowLength, windowShift);
			vis.computeAndRepaintSpectrogram();
			vs.setWinType("hann");
			vs.setWinLen(windowLength);
			try {
				vs.update();
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			

		}	
		else if (e.getActionCommand() == "Rectangle") {
			windowType = "rect";
			vis.setWindowConfig(windowType, windowLength, windowShift);
			vis.computeAndRepaintSpectrogram();
			vs.setWinType("rect");
			vs.setWinLen(windowLength);
			try {
				vs.update();
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			
		}			
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		JSlider slider = (JSlider) e.getSource();
		if (slider.getToolTipText() == "brightness") {
			vis.setBrightness(1.f - (float)slider.getValue() / (float)slider.getMaximum());
			vis.repaintAndUpdateImage();
			return;
		} else if (slider.getToolTipText() == "contrast") {
			vis.setContrast(((double)slider.getMaximum() - (double)slider.getValue())/10.);
			vis.repaintAndUpdateImage();
			return;
		}
		if (!slider.getValueIsAdjusting()) {
			if (slider.getToolTipText() == "windowShift") {
				windowShift = slider.getValue();
				labelShift.setText("Shift: " + windowShift);
			} else if (slider.getToolTipText() == "windowLength") {
				windowLength = slider.getValue();
				labelLength.setText("Length: " + windowLength);
				vs.setWinLen(windowLength);
				try {
					vs.update();
				} catch (Exception e1) {
					e1.printStackTrace();
				}
			} 
			vis.setWindowConfig(windowType, windowLength, windowShift);
			vis.computeAndRepaintSpectrogram();
		}
	}

	@Override
	public void itemStateChanged(ItemEvent e) {
		Object source = e.getItemSelectable();
		if (source == checkBoxAveraged) {
			if (e.getStateChange() == ItemEvent.DESELECTED) vis.useAverage(false);
			if (e.getStateChange() == ItemEvent.SELECTED) vis.useAverage(true);
			vis.repaintAndUpdateImage();
		} else if (source == checkBoxColored) {
			if (e.getStateChange() == ItemEvent.DESELECTED) {
				vis.setColored(false);
				vis.setColorScheme(new Color(255, 255, 255), new Color(255, 255, 0, 100), new Color(0, 0, 0), new Color(255, 0, 0));

			}
			if (e.getStateChange() == ItemEvent.SELECTED) {
				vis.setColored(true);
				vis.setColorScheme(new Color(0, 0, 0), new Color(100, 100, 100, 50), new Color(0, 0, 0), new Color(100, 100, 100));
			}

			vis.repaintAndUpdateImage();
			
		} else if (source == checkBoxLog) {
			if (e.getStateChange() == ItemEvent.DESELECTED) vis.setLog(false);
			if (e.getStateChange() == ItemEvent.SELECTED) vis.setLog(true);

			vis.repaintAndUpdateImage();
		} else if (source == checkBoxFrameDiagrams) {
			if (e.getStateChange() == ItemEvent.SELECTED) diagrams.setVisible(true);
			if (e.getStateChange() == ItemEvent.DESELECTED) diagrams.setVisible(false);
		}
	}
}
