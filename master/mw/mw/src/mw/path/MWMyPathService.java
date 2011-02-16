package mw.path;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import javax.jws.soap.SOAPBinding;

import mw.cache.MWCacheClient;
import mw.cache.MWNoSuchKeyException;
import mw.facebook.*;


@javax.jws.WebService(name = "MWMyPathService", serviceName = "MWPathService")
@javax.jws.soap.SOAPBinding(style = SOAPBinding.Style.RPC)

public class MWMyPathService implements MWPathServiceInterface 
{
	private MWMyFacebookService facebook; // proxy
	private MWCacheClient cache;
	final int MAX_ITER = 100;
	int VERBOSITY = 0;

	public MWMyPathService()
	{
		MWFacebookService service = new MWFacebookService();
		facebook = service.getMWMyFacebookServicePort();
		cache = new MWCacheClient();
	}
	
	private Collection<String> getFriends(String id)
	{
		Collection<String> friends = new ArrayList<String>();
		
		Map<String, String> m = null;
		try {
			m = cache.getBucket(id);
		} catch (MWNoSuchKeyException e1) {
			if (VERBOSITY > 1)
				System.out.println(id + " not available in cache so far.");
		}
		
		if (m != null && m.size() > 0) {
			if (m.size() > 0) {
				for (String str : m.keySet()) {
					friends.add(str);
				}
			}
		} else {
			try {
				StringArray sa = facebook.getFriends(id);
				cache.addBucket(id);
				for (String s: sa.getItem())
				{
					friends.add(s);
					cache.addObject(id+"/"+s, "#");
				}

			} catch (MWUnknownIDException_Exception e) {
				if (VERBOSITY > 0) {
					System.err.println("Fehler bei Aufruf von getFriends(). ID ungueltig?");
					System.err.println(e.getMessage());
				}
			}
		}

		return friends;
	}
	
	private Collection<String> concat(Collection<String> a, Collection<String> b) 
	{
		for (String str: b) 
			if (!a.contains(str)) 
				a.add(str);
		return a;
	}
	
	protected String[] tryPathFromCache(String startID, String endID) {
		String pathFromCache = null;
		try {
			pathFromCache = cache.getObject(startID+","+endID);
		} catch (MWNoSuchKeyException e) {
			if (VERBOSITY > 1)
				System.out.println("Path from " + startID + " to " + endID + " not in cache so far.");
		}
		
		if (pathFromCache != null)
		{
			StringTokenizer st = new StringTokenizer(pathFromCache, ",");
			String[] path = new String[st.countTokens()];
			int i = 0;
			while (st.hasMoreTokens()) {
				path[i++] = st.nextToken();
			}
			return path;
		} else {
			return null;
		}
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	@javax.jws.WebMethod
	public String[] calculatePath(String startID, String endID) throws MWNoPathException {	
		
		String[] pathFromCache = tryPathFromCache(startID, endID);
		if (pathFromCache != null)
			return pathFromCache;
		
		
		int aufruf = 0;
		Collection<String> graph1 = new ArrayList<String>();
		Collection<String> graph2 = new ArrayList<String>();
		Collection<String> graph1_new = new ArrayList<String>();
		Collection<String> graph2_new = new ArrayList<String>();
		Collection<String> graph1_cache = new ArrayList<String>();
		Collection<String> graph2_cache = new ArrayList<String>();
		
		StringArray arr = new StringArray();
		StringArray initial = new StringArray();
		StringArrayArray init_aa = null;
		List<String> initial_list = initial.getItem();
		initial_list.add(startID);
		initial_list.add(endID);
		
		long startTimeGesamt = System.currentTimeMillis();
		
		try {
			init_aa = facebook.getFriendsBatch(initial); aufruf++;
		} catch (MWUnknownIDException_Exception e1) {
			if (VERBOSITY > 0)
				e1.printStackTrace();
		}
			
		graph1.add(startID);
		graph2.add(endID);
		cache.addBucket(startID);
		cache.addBucket(endID);
		
		StringArray sa = init_aa.getItem().get(0);		
		for (String s: sa.getItem())
		{
			graph1_new.add(s);
			cache.addObject(startID + "/" + s, "#");
		}
		sa = init_aa.getItem().get(1);
		for (String s: sa.getItem())
		{
			graph2_new.add(s);
			cache.addObject(endID + "/" + s, "#");
		}
		
		
		for (int i = 1; true; i++)
		{
			long startTime = System.currentTimeMillis();
			
			if (i > MAX_ITER)
				throw new MWNoPathException();
			
			if (!Collections.disjoint(concat(graph1,graph1_new), concat(graph2,graph2_new))) 
				break;
			
			List<String> tmparr = arr.getItem();
			tmparr.clear();
			
			int elems = 0;	
			for (String str : graph1_new) {	
				Map<String, String> m = null;
				try {
					m = cache.getBucket(str);
				} catch (MWNoSuchKeyException e1) {
					if (VERBOSITY > 1)
						System.out.println(str + " not available in cache so far.");
				}
				
				if (m != null && m.size() > 0) {
					for (String str2 : m.keySet()) {
						graph1_cache.add(str2);
					}
				} else {
					tmparr.add(str);
					++elems;
				}
			}
			for (String str : graph2_new) {
				Map<String, String> m = null;
				try {
					m = cache.getBucket(str);
				} catch (MWNoSuchKeyException e1) {
					if (VERBOSITY > 1)
						System.out.println(str + " not available in cache so far.");
				}
				
				if (m != null && m.size() > 0) {
					for (String str2 : m.keySet()) {
						graph2_cache.add(str2);
					}
				} else {
					tmparr.add(str);
				}
			}
			
			try {
				init_aa = facebook.getFriendsBatch(arr); aufruf++;
			} catch (MWUnknownIDException_Exception e1) {
				if (VERBOSITY > 0)
					e1.printStackTrace();
			}
			
			graph1 = concat(graph1, graph1_new);
			graph2 = concat(graph2, graph2_new);
			graph1_new = concat(new ArrayList(), graph1_cache);
			graph2_new = concat(new ArrayList(), graph2_cache);
			graph1_cache.clear();
			graph2_cache.clear();
			
			int j = 0;
			for( StringArray stra : init_aa.getItem()) {
				cache.addBucket(tmparr.get(j));
				if(j < elems) {
					for (String str: stra.getItem()) {
						cache.addObject(tmparr.get(j) + "/" + str, "#");
						if (!graph1_new.contains(str)) {
							graph1_new.add(str);			
						}	
					}
				} else {
					for (String str: stra.getItem()) {
						cache.addObject(tmparr.get(j) + "/" + str, "#");
						if (!graph2_new.contains(str)) {
							graph2_new.add(str);			
						}	
					}
				}
				j++;
			}
				
			long endTime = System.currentTimeMillis();
			if (VERBOSITY > 1)
				System.out.println("Iteration " + i + " " + (endTime - startTime) + " ms");
		}
		
		
		long endTimeGesamt = System.currentTimeMillis();
		if (VERBOSITY > 1)
			System.out.println("Gesamtzeit " + (endTimeGesamt - startTimeGesamt) + " ms (Aufrufe: " + aufruf + ")");
		
		// send path to cache
		String[] result = MWDijkstra.getShortestPath(facebook, concat(graph1, graph2), startID, endID);
		String wholePath = new String();
		for (int i = 0; i < result.length-1; i++) {
			wholePath += result[i] + ",";
		}
		wholePath += result[result.length-1];
		cache.addObject(startID+","+endID, wholePath);
		
		return result;
	}
	
	@javax.jws.WebMethod
	public String[] calculatePathSlow(String startID, String endID) throws MWNoPathException {
		
		String[] pathFromCache = tryPathFromCache(startID, endID);
		if (pathFromCache != null)
			return pathFromCache;
		
		
		// OR do that... muhahaha
		int aufruf = 0;
		long startTimeGesamt = System.currentTimeMillis();
		
		Collection<String> graph1 = getFriends(startID); aufruf++;
		Collection<String> graph2 = getFriends(endID); aufruf++;
		
		for (int i = 1; true; i++)
		{
			long startTime = System.currentTimeMillis();
			
			if (i > MAX_ITER)
			{
				throw new MWNoPathException();
			}
			
			if (!Collections.disjoint(graph1, graph2)) 
				break;
			
			Collection<String> tmp = new ArrayList<String>();
			for (String str: graph1)
			{
				tmp = concat(tmp, getFriends(str)); aufruf++;
			}
			graph1 = concat(graph1, tmp);
			tmp.clear();
			
			for (String str: graph2)
			{
				tmp = concat(tmp, getFriends(str)); aufruf++;
			}
			graph2 = concat(graph2, tmp);
			long endTime = System.currentTimeMillis();
			if (VERBOSITY > 1)
				System.out.println("Iteration " + i + " " + (endTime - startTime) + " ms");
		}
		
		long endTimeGesamt = System.currentTimeMillis();
		if (VERBOSITY > 1)
			System.out.println("Gesamtzeit " + (endTimeGesamt - startTimeGesamt) + " ms (Aufrufe: " + aufruf + ")");
		
		String[] result = MWDijkstra.getShortestPath(facebook, concat(graph1, graph2), startID, endID);
		
		String wholePath = new String();
		for (int i = 0; i < result.length-1; i++) {
			wholePath += result[i] + ",";
		}
		wholePath += result[result.length-1];
		cache.addObject(startID+","+endID, wholePath);
		
		return result;
	}
	
	public static void main(String[] args) throws MWNoPathException
	{
		MWMyPathService svc = new MWMyPathService();
		
		String str = "1694452301";
		String str2 = "100000859170147";
		System.out.println("calculatePath(" + str + ", " + str2 + ")");
		long startTimeGesamt = System.currentTimeMillis();
		svc.calculatePath(str, str2);
		long endTimeGesamt = System.currentTimeMillis();
		System.out.println("\nPfadberechnung IMPROVED Dauer: " + (endTimeGesamt-startTimeGesamt) + " ms");
		System.out.println("----------------------------------------------");
		System.out.println("calculatePathSlow(" + str + ", " + str2 + ")");
		long startTimeGesamt2 = System.currentTimeMillis();
		svc.calculatePathSlow(str, str2);
		long endTimeGesamt2 = System.currentTimeMillis();
		System.out.println("\nPfadberechnung SLOW Dauer: " + (endTimeGesamt2-startTimeGesamt2) + " ms");
	}
}
