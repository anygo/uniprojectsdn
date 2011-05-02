package vsue.tests;

public interface VSRemoteObject extends VSRemote {
	public String getString();
	public String concat(String string);
	public String concat(String pre, String post);
	public void setString(String string);
}
