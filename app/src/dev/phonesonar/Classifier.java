package dev.phonesonar;

/** Something that can say, from one echo, how likely it is to have come from each object. */
interface Classifier {
    String[] CLASSES = {"fish", "bobblehead"};

    /**
     * @param x one example: [4 pairs][delays], each value 0..1
     * @return how likely each class is, adding up to 1, in the order of CLASSES
     */
    float[] classify(float[][] x);
}
