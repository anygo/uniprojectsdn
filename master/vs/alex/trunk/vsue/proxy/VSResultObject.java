package vsue.proxy;

import java.io.Serializable;

public class VSResultObject implements Serializable {
	
	private Object m_result;
	
	public VSResultObject(Object result) {
		
		m_result = result;
		
	}
	
	public Object getObject() {
		
		return m_result;
	
	}

}
