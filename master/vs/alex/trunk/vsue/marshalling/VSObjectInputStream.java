package vsue.marshalling;

import java.util.Vector;
import java.io.InputStream;
import java.io.NotSerializableException;
import java.io.Serializable;
import java.lang.reflect.*;
import java.io.IOException;
import java.lang.ClassNotFoundException;
import java.nio.*;
import sun.reflect.ReflectionFactory;
import java.security.AccessController;

public class VSObjectInputStream {

	private Vector<Object> m_brain;
	private InputStream m_in;

	public VSObjectInputStream(InputStream in) {

		m_brain = new Vector<Object>();
		m_in = in;

	}

	public InputStream getInputStream() {

		return m_in;

	}

	public Object readObject() throws IOException, ClassNotFoundException {

		/**
		 * 1)   Check control byte
		 *        (byte) 0 : reset m_brain
		 *        (byte) 1 : normal operation
		 *
		 * 2)   Check int start
		 * 2.1)   start == 0 : object should point to null
		 * 2.2)   start < 0  : object is in m_brain[(start * -1) - 1]
		 * 2.3)   start > 0  : the next 'start' bytes after start contain a classname
		 */

		/**
		 * 1)   Check control byte
		 */
		while(getByte() == (byte)0) {
			m_brain.clear();
		}

		/**
		 * 2)   Check int start
		 */
		int start = getInt();
		if(start == 0) {

			/**
		 	 * 2.1)   start == 0 : object should point to null
			 */
			return null;

		} else if (start < 0) {

			/**
		 	 * 2.2)   start < 0  : object is in m_brain[(start * -1) - 1]
			 */
			return m_brain.get((start * -1) -1);

		} else {

			/**
		 	 * 2.3)   start > 0  : the next 'start' bytes after start contain a classname
			 */
			ByteBuffer tmpbuf = getByteBuffer(start);
			String classname = new String(tmpbuf.array());
			Class objClass = getClassObject(classname);
			Object object = null;

			// Not serializable check
			if(!Serializable.class.isAssignableFrom(objClass)) {
				throw new NotSerializableException();
			}
			
			if(objClass.isArray()) {

				int len = getInt();
				object = Array.newInstance(objClass.getComponentType(), len);
				m_brain.add(object);

				for(int i = 0; i < len; i++) {
					Array.set(object, i, readObject());
				}

			} else {

				try {

					// TODO handle exceptions?
					// java.lang.InstantiationException,
					// java.lang.IllegalAccessException
					// ExceptionInInitializerError
					// SecurityException
					Class c = objClass;
					while (java.io.Serializable.class.isAssignableFrom(c)) {
						c = c.getSuperclass();
						if (c == null) throw new Exception();
					}
					Constructor instConstr = c.getConstructor(new Class[0]);
					ReflectionFactory reflFactory =
						(ReflectionFactory)AccessController.doPrivileged(
							new ReflectionFactory.GetReflectionFactoryAction());
					Constructor constructor =
						reflFactory.newConstructorForSerialization(objClass,
											instConstr);
					object = constructor.newInstance((Object[]) null);
					m_brain.add(object);

				} catch (Exception e) {
					e.printStackTrace();
				}

				// Don't handle VSExternalizable objects here. Call
				// writeExternal instead
				if(VSExternalizable.class.isAssignableFrom(objClass)) {

					((VSExternalizable)object).readExternal(this);

				} else {

					Field[] attributes = objClass.getDeclaredFields();
					for(int i = 0; i < attributes.length; i++) {

						Field attr = attributes[i];
						Class c = attr.getType();
						int m = attr.getModifiers();
						boolean restoreAccess = false;

						// skip if transient or static final
						if(Modifier.isTransient(m) ||
						   (Modifier.isStatic(m) &&
						    Modifier.isFinal(m)) ) {

						     continue;
						}

						if(!attr.isAccessible()) {
							restoreAccess = true;
							attr.setAccessible(true);
						}

						try {
							if(c.isPrimitive()) {

								if(c == boolean.class) {
									attr.setBoolean(object, getBoolean());
								} else if(c == byte.class) {
									attr.setByte(object, getByte());
								} else if(c == char.class) {
									attr.setChar(object, getChar());
								} else if(c == double.class) {
									attr.setDouble(object, getDouble());
								} else if(c == float.class) {
									attr.setFloat(object, getFloat());
								} else if(c == int.class) {
									attr.setInt(object, getInt());
								} else if(c == long.class) {
									attr.setLong(object, getLong());
								} else { // short.class
									attr.setShort(object, getShort());
								}

							} else { // -> not primitive
								attr.set(object, readObject());
							}

						} catch (IllegalAccessException e) {
							e.printStackTrace();
						}

						if(restoreAccess) {
							attr.setAccessible(false);
						}
					}
				}
			}

			return object;
		}
	}

	public void close() throws IOException {

		m_brain.clear();
		m_in.close();

	}

	private ByteBuffer getByteBuffer(int size) throws IOException {
		int start, res;
		int bytepos = 0;
		byte[] byteArr = new byte[size];
		while(bytepos < size) {
			res = m_in.read(byteArr, bytepos, size - bytepos);
			if(res > 0) {
				bytepos += res;
			} else {
				throw new IOException();
			}
		}
		return ByteBuffer.wrap(byteArr);
	}

	private boolean getBoolean() throws IOException {
		byte ret = getByteBuffer(Byte.SIZE/8).get();
		if(ret == (byte)1) {
			return true;
		} else {
			return false;
		}
	}

	private byte getByte() throws IOException {
		return getByteBuffer(Byte.SIZE/8).get();
	}

	private char getChar() throws IOException {
		return getByteBuffer(Character.SIZE/8).getChar();
	}

	private double getDouble() throws IOException {
		return getByteBuffer(Double.SIZE/8).getDouble();
	}

	private float getFloat() throws IOException {
		return getByteBuffer(Float.SIZE/8).getFloat();
	}

	private int getInt() throws IOException {
		return getByteBuffer(Integer.SIZE/8).getInt();
	}

	private long getLong() throws IOException {
		return getByteBuffer(Long.SIZE/8).getLong();
	}

	private short getShort() throws IOException {
		return getByteBuffer(Short.SIZE/8).getShort();
	}

	private Class getClassObject(String name) throws ClassNotFoundException {
		try {
			return Class.forName(name);
		} catch (Exception e) {
			throw new ClassNotFoundException();
		}
	}

}

