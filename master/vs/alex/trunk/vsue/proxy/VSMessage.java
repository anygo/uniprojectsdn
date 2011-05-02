package vsue.proxy;

import java.io.Serializable;

public class VSMessage implements Serializable {

	private Object m_transportObject;
	
	public VSMessage(Object obj) {

		m_transportObject = obj;

	}

	// TODO: use isXXObject()
	public Exception getException() {
		
		if(m_transportObject != null && Exception.class.isAssignableFrom(m_transportObject.getClass())) {
			return (Exception)m_transportObject;
		}

		return null;

	}
	
	public VSLookupObject getVSLookupObject() {
		
		if(m_transportObject != null && VSLookupObject.class.isAssignableFrom(m_transportObject.getClass())) {
			return (VSLookupObject)m_transportObject;
		}

		return null;

	}
	
	public VSMethodObject getVSMethodObject() {
		
		if(m_transportObject != null && VSMethodObject.class.isAssignableFrom(m_transportObject.getClass())) {
			return (VSMethodObject)m_transportObject;
		}

		return null;		
		
	}
	
	public VSRemoteReference getVSRemoteReference() {
		
		if(m_transportObject != null && VSRemoteReference.class.isAssignableFrom(m_transportObject.getClass())) {
			return (VSRemoteReference)m_transportObject;
		}

		return null;
		
	}
	
	public VSResultObject getVSResultObject() {
		
		if(m_transportObject != null && VSResultObject.class.isAssignableFrom(m_transportObject.getClass())) {
			return (VSResultObject)m_transportObject;
		}

		return null;
		
	}

}
