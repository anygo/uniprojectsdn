package vs;

import java.io.Serializable;

public class Message implements Serializable {

	private int m_uid;
	private String m_title;
	private String m_message;

	public Message(int uid, String title, String message) {
		m_uid = uid;

		// create new strings and copy the data
		m_title = title;
		m_message = message;
	}

	public void setUID(int uid) {
		m_uid = uid;
	}
	public void setTitle(String title) {
		m_title = title;
	}
	public void setMessage(String message) {
		m_message = message;
	}

	public int getUID() {
		return m_uid;
	}
	public final String getTitle() {
		return m_title;
	}
	public final String getMessage() {
		return m_message;
	}

	public void print() {
		System.out.println(toString());
	}
	public String toString() {
		return 	m_uid + " '" +
			m_title + "' '" +
			m_message + "'";
	}
}

