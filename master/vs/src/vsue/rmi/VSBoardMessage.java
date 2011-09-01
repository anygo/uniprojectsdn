package vsue.rmi;

import java.io.Serializable;

public class VSBoardMessage implements Serializable {

	private static final long serialVersionUID = 1L;

	private int uid;
	private String title;
	private String message;

	public int getUid() {
		return uid;
	}

	public void setUid(int uid) {
		this.uid = uid;
	}

	public String getTitle() {
		return title;
	}

	public void setTitle(String title) {
		this.title = title;
	}

	public String getMessage() {
		return message;
	}

	public void setMessage(String message) {
		this.message = message;
	}

	public String toString() {
		return "Message (" + getUid() + ", \"" + getTitle() + "\", \""
				+ getMessage() + "\")";
	}
}
