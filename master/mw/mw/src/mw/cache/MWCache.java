package mw.cache;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBElement;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.util.JAXBSource;
import javax.xml.transform.Source;
import javax.xml.ws.Endpoint;
import javax.xml.ws.Provider;
import javax.xml.ws.Service;
import javax.xml.ws.ServiceMode;
import javax.xml.ws.WebServiceContext;
import javax.xml.ws.WebServiceProvider;
import javax.xml.ws.handler.MessageContext;
import javax.xml.ws.http.HTTPBinding;

import mw.cache.generated.MWHash;
import mw.cache.generated.ObjectFactory;

@WebServiceProvider
@ServiceMode(value=Service.Mode.PAYLOAD)
public class MWCache implements Provider<Source> {

	@javax.annotation.Resource(type=WebServiceContext.class)
	protected WebServiceContext wsContext;

	static protected Map<String, String> map;
	static protected Map<String, Map<String, String> > buckets;
	static volatile boolean doBackUp = false;
	protected int write;
	protected int read;
	protected int miss;
	int VERBOSITY = 0;
	
	
	static private void restore() {
		
		try {
			BufferedReader br = new BufferedReader(new FileReader("map.dat"));
			
			String line = br.readLine();
			while (line != null) {
				StringTokenizer st = new StringTokenizer(line, " ");
				map.put(st.nextToken(), st.nextToken());
				line = br.readLine();
			}	
		} catch (Exception e) {
			System.out.println("map.dat not found. cache empty.");
		}
		
		try {
			BufferedReader br = new BufferedReader(new FileReader("buckets.dat"));
			
			String line = br.readLine();
			while (line != null) {
				StringTokenizer st = new StringTokenizer(line, " ");
				String bucketKey = st.nextToken();
				HashMap<String, String> hm = new HashMap<String, String>();
				
				while(st.hasMoreTokens()) {
					hm.put(st.nextToken(), st.nextToken());
				}
				buckets.put(bucketKey, hm);
				line = br.readLine();
			}	
		} catch (Exception e) {
			System.out.println("buckets.dat not found. cache empty.");
		}
	}
	
	static private void backup() {
		
		// backup map
		try {
			FileWriter fMap = new FileWriter(new File("map.dat"));
			BufferedWriter outMap = new BufferedWriter(fMap);
			Set<String> allKeys = map.keySet();
			synchronized (map) {
				for (String str : allKeys) {
					outMap.write(str + " " + map.get(str) + "\n");
				}
			}
			outMap.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// backup buckets
		try {
			FileWriter fMap = new FileWriter(new File("buckets.dat"));
			BufferedWriter outMap = new BufferedWriter(fMap);
			Set<String> allKeys = buckets.keySet();
			synchronized (buckets) {
				for (String bucket : allKeys) {
					Map<String, String> hm = buckets.get(bucket);
					Set<String> allBucketKeys = hm.keySet();
					outMap.write(bucket + " ");
					
					for (String object : allBucketKeys) {
						outMap.write(object +  " " + hm.get(object) + " ");
					}
					outMap.write("\n");
				}
			}
			outMap.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public MWCache()
	{
		Map<String, String> mapTmp = new HashMap<String, String>();
		map = Collections.synchronizedMap(mapTmp);
		Map<String, Map<String, String> > bucketsTmp = new HashMap<String, Map<String, String> >();
		buckets = Collections.synchronizedMap(bucketsTmp);
		restore();
	}

	public Source invoke(Source source) {
		
		//System.out.println("map: " + map.size() + "; buckets: " + buckets.size());
		
		MessageContext mc = wsContext.getMessageContext();
		String httpMethod = (String)mc.get(MessageContext.HTTP_REQUEST_METHOD);
		String pathInfo = (String)mc.get(MessageContext.PATH_INFO);
		
		String contextPath = "mw.cache.generated";
		MWHash input = null;
		JAXBContext jc = null;
		try {
			jc = JAXBContext.newInstance(contextPath);
		} catch (JAXBException e) {
			e.printStackTrace();
		}
		if (httpMethod.equalsIgnoreCase("GET")) {
			
			ObjectFactory f = new ObjectFactory();
			MWHash hash = f.createMWHash();
			hash.setException(false);
			
			if (pathInfo.contains("/")) { // getObject() called (cache/bucketKey/objectKey)
				StringTokenizer st = new StringTokenizer(pathInfo, "/");
				String bucketKey = st.nextToken();
				String objectKey = st.nextToken();
				if (buckets.containsKey(bucketKey)) {
					Map<String, String> bucket = Collections.synchronizedMap(buckets.get(bucketKey));
					if (bucket.containsKey(objectKey)) {
						List<String> list = hash.getList(); // list contains only ONE element in this case
						list.add(bucket.get(objectKey));
					}
				} else { 
					// exception ...
					hash.setException(true);
				}
			} else { // getObject() OR getBucket called -> check!
				
				if (map.containsKey(pathInfo)) {
					List<String> list = hash.getList();
					list.add(map.get(pathInfo));
				} else if (buckets.containsKey(pathInfo)) {
					List<String> list = hash.getList();
					Map<String, String> hashMap = buckets.get(pathInfo);
					Set<String> allKeys = hashMap.keySet();
					synchronized (hashMap) {
						for (String str : allKeys) { // key, value, key, value, key, value, ...
							list.add(str);
							list.add(hashMap.get(str));
						}
					}
					
				} else {
					// exception
					hash.setException(true);
				}
				
			}
			JAXBElement<MWHash> reply = f.createMWCacheReply(hash);
			try {
				Source replySource = new JAXBSource(jc, reply);
				return replySource;
			} catch (JAXBException e) {
				e.printStackTrace();
			}
		
			
		} else if (httpMethod.equalsIgnoreCase("POST")) { // 	POST-method
			
			List<String> val = null;
			String method = null;
			try {
				Unmarshaller u = jc.createUnmarshaller();
				@SuppressWarnings("rawtypes")
				JAXBElement request = (JAXBElement) u.unmarshal(source);
				input = (MWHash)request.getValue();
				method = input.getMethod();
				val = input.getList();
			} catch (JAXBException e1) {
				e1.printStackTrace();
			}
			
			if (method.equalsIgnoreCase("addBucket")) {
				synchronized (this) {
					if (map.containsKey(pathInfo))
					{
						if (VERBOSITY > 1)
							System.out.println("objectKey already exists - no bucket created!");
					} else {
						buckets.put(pathInfo, new HashMap<String, String>());	
					}
				}
			} else if (method.equalsIgnoreCase("addObject")) {
				if (pathInfo.contains("/")) {
					StringTokenizer st = new StringTokenizer(pathInfo, "/");
					String bucketKey = st.nextToken();
					String objectKey = st.nextToken();
					if (!buckets.containsKey(bucketKey)) {
						// Do nothing - error... bla
						if (VERBOSITY > 1)
							System.out.println("Bucket not existing ");
					} else {
						buckets.get(bucketKey).put(objectKey, val.get(0));
					}
				} else { // flat inputting
					map.put(pathInfo, val.get(0));
				}
			}
			doBackUp = true;
			
		}
		
		return null;
	}
	
	public static void main(String args[]) {
		Endpoint endpoint = Endpoint.create(HTTPBinding.HTTP_BINDING, new MWCache());
		System.out.println("Cache Server initialized!");
		
		// Backup Thread
		new Thread(new Runnable() {
			
			@Override
			public void run() {
				while (true) {
					try {
						Thread.sleep(500);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
					if (doBackUp)
						
						backup();
					doBackUp = false;
				}
				
			}
		}).start();
		
		endpoint.publish("http://localhost:42042/cache/");	
	}

}
