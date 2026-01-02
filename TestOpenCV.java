public class TestOpenCV {
}
package com.example;

import org.opencv.core.Core;

public class TestOpenCV {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        System.out.println("OpenCV version: " + Core.VERSION);
    }
}
