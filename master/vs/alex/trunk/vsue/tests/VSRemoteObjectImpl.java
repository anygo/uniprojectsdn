package vsue.tests;


public class VSRemoteObjectImpl implements VSRemoteObject {
	private String string;
	
	public String getString() {
		return string;
	}

	public String concat(String string) {
		return (this.string.concat(string));
	}
	
	public String concat(String pre, String post) {
		return (pre.concat(this.string.concat(post)));
	}
	
	public void setString(String string) {
		this.string = string;
	}

}
