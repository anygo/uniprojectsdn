package mw.cache;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBElement;
import javax.xml.bind.JAXBException;
import javax.xml.namespace.QName;
import javax.xml.ws.Dispatch;
import javax.xml.ws.Service;
import javax.xml.ws.handler.MessageContext;
import javax.xml.ws.http.HTTPBinding;
import mw.cache.generated.MWHash;
import mw.cache.generated.ObjectFactory;

public class MWCacheClient {

	private final String CACHEPATH = "http://localhost:42042/cache/";
	int VERBOSITY = 0;
	
	public void addBucket(String key)
	{
		String path = CACHEPATH+key;
		QName qName = new QName("", "");
		Service service = Service.create(qName);
		
		service.addPort(qName, HTTPBinding.HTTP_BINDING, path);
		
		String contextPath = "mw.cache.generated";
		try {
			JAXBContext jc = JAXBContext.newInstance(contextPath);
			Dispatch<Object> dispatch = service.createDispatch(qName, jc, Service.Mode.PAYLOAD);
			Map<String, Object> rc = dispatch.getRequestContext();
			rc.put(MessageContext.HTTP_REQUEST_METHOD, "POST");
			
			ObjectFactory f = new ObjectFactory();
			
			MWHash requestValue = f.createMWHash();
			requestValue.setMethod("addBucket");
			
			JAXBElement<MWHash> request = f.createMWCacheRequest(requestValue);
			dispatch.invokeOneWay(request);
			
			if (VERBOSITY > 1)
				System.out.println("Bucket created.");
	 		
		} catch (JAXBException e) {
			e.printStackTrace();
		}
	}
	
	public Map<String, String> getBucket(String key) throws MWNoSuchKeyException
	{
		String path = CACHEPATH+key;
		QName qName = new QName("", "");
		Service service = Service.create(qName);
		
		service.addPort(qName, HTTPBinding.HTTP_BINDING, path);
		
		String contextPath = "mw.cache.generated";
		try {
			JAXBContext jc = JAXBContext.newInstance(contextPath);
			Dispatch<Object> dispatch = service.createDispatch(qName, jc, Service.Mode.PAYLOAD);
			Map<String, Object> rc = dispatch.getRequestContext();
			rc.put(MessageContext.HTTP_REQUEST_METHOD, "GET");
			
			ObjectFactory f = new ObjectFactory();

			JAXBElement<MWHash> request = f.createMWCacheRequest(null);
			@SuppressWarnings("rawtypes")
			JAXBElement reply = (JAXBElement) dispatch.invoke(request);
			MWHash replyValue = (MWHash)reply.getValue();
			List<String> list = replyValue.getList();
			
			if (replyValue.isException())
				throw new MWNoSuchKeyException();
			
			HashMap<String, String> hm = new HashMap<String, String>();
			for(int i = 0; i < list.size(); i += 2) {	
				hm.put(list.get(i), list.get(i+1));
			}
	 		
			return hm;
			
		} catch (JAXBException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public void addObject(String key, String value)
	{
		String path = CACHEPATH+key;
		QName qName = new QName("", "");
		Service service = Service.create(qName);
		
		service.addPort(qName, HTTPBinding.HTTP_BINDING, path);
		
		String contextPath = "mw.cache.generated";
		try {
			JAXBContext jc = JAXBContext.newInstance(contextPath);
			Dispatch<Object> dispatch = service.createDispatch(qName, jc, Service.Mode.PAYLOAD);
			Map<String, Object> rc = dispatch.getRequestContext();
			rc.put(MessageContext.HTTP_REQUEST_METHOD, "POST");
			
			ObjectFactory f = new ObjectFactory();
			
			MWHash requestValue = f.createMWHash();
			requestValue.setMethod("addObject");
			requestValue.getList().add(value);
			
			JAXBElement<MWHash> request = f.createMWCacheRequest(requestValue);
			dispatch.invokeOneWay(request);
			if (VERBOSITY > 1)
				System.out.println("Schluessel-Wert Paar gesendet.");
	 		
		} catch (JAXBException e) {
			e.printStackTrace();
		}
	}
	
	public String getObject(String key) throws MWNoSuchKeyException
	{
		String path = CACHEPATH+key;
		QName qName = new QName("", "");
		Service service = Service.create(qName);
		
		service.addPort(qName, HTTPBinding.HTTP_BINDING, path);
		
		String contextPath = "mw.cache.generated";
		try {
			JAXBContext jc = JAXBContext.newInstance(contextPath);
			Dispatch<Object> dispatch = service.createDispatch(qName, jc, Service.Mode.PAYLOAD);
			Map<String, Object> rc = dispatch.getRequestContext();
			rc.put(MessageContext.HTTP_REQUEST_METHOD, "GET");
			
			ObjectFactory f = new ObjectFactory();

			JAXBElement<MWHash> request = f.createMWCacheRequest(null);
			@SuppressWarnings("rawtypes")
			JAXBElement reply = (JAXBElement) dispatch.invoke(request);
			MWHash replyValue = (MWHash)reply.getValue();
			
			if (replyValue.isException()) 
				throw new MWNoSuchKeyException();

			if (VERBOSITY > 1)
				System.out.println(replyValue.getList().get(0));
	 		
			return replyValue.getList().get(0);
			
		} catch (JAXBException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static void main(String[] args) {
		
		MWCacheClient cache = new MWCacheClient();
		try {
			cache.getObject("hoden");
			cache.getObject("auto/mercedes");
		} catch (MWNoSuchKeyException e) {
			e.printStackTrace();
		}

	}
}
