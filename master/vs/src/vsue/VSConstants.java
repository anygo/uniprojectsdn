package vsue;

public class VSConstants {
	public static enum RPC_SEMANTICS_ENUM { LAST_OF_MANY, AT_MOST_ONCE, MAYBE };
	
	public final static String CLOSED_MESSAGE = "Client closed connection or EOF reached";
	
	public static final int REGISTRY_PORT_BOARD_SERVER = 12345;
	
	public static final int VSSERVER_STARTING_PORT = 12346;
	
	public static int SEND_MULTIPLE_TIME_MAX_ATTEMPTS = 3;
	
	public static int SEND_MULTIPLE_TIME_WAITING_TIME = 3000;
	
	public static int JGROUPS_GET_STATE_TIMEOUT = 0;
	
	public static RPC_SEMANTICS_ENUM RPC_SEMANTICS = RPC_SEMANTICS_ENUM.MAYBE;	
	
	public static String REPLICA_1 = "faui08k";
	
	public static String REPLICA_2 = "faui08j";

	public static String REPLICA_3 = "faui08l";
	
	public static String VSBOARD_SERVER_NAME = REPLICA_1;
	
	public static String GROUP_NAME = "gruppe1";
	
	public static boolean OPTIMIZED_ACK_STRATEGY = false;
}
