package vsue.replica;

public interface VSRemoteObjectStateHandler {
	public byte[] getState();
	public void setState(byte[] state);
}
