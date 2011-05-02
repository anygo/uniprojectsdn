package vsue.proxy;

import java.io.Serializable;
import java.lang.reflect.Method;

public class VSMethodObject implements Serializable {

	private int m_objectID;
	private String m_genericMethodName;
	private Object[] m_args;
	
	public VSMethodObject(int objectID, Method method, Object[] args) {
	
		m_objectID = objectID;
		m_genericMethodName = method.toGenericString();
		m_args = args;
		
	}
	
	public int getObjectID() {
		
		return m_objectID;
		
	}
	
	public String getGenericMethodName() {
		
		return m_genericMethodName;
		
	}
	
	public Object[] getArgs() {
		
		return m_args;
		
	}
	
}
