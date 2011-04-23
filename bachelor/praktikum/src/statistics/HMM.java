package statistics;

/**
 * Abstract HMM class. Covers properties shared by all HMM topologies
 * @author sikoried
 *
 */
public abstract class HMM {
	/** unique id */
	int id;
	
	/** transition probabilities */
	double [] trans;
	
	/** number of states */
	int ns;	
}
