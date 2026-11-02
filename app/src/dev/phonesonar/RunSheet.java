package dev.phonesonar;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * What to record in a block, and in what order. A block visits every position on the mat once, in a
 * shuffled order, and ends by repeating the first, which shows how far the block has drifted. Each set is
 * the empty scene and then both objects at that position, the one to go first chosen at random. About one
 * set in five, and at least one in a block, also has an empty trial, somewhere after the empty scene: hands
 * reach in and nothing stays, which shows what a disturbed scene looks like with nothing there.
 *
 * The same seed always gives the same sheet, so a block can be picked up again if the app is closed.
 */
final class RunSheet {
    static final int[] DISTANCES = {10, 12, 14, 16, 18, 20};   // cm, back of the case to the centre of the object's base
    static final int[] SIDEWAYS = {0};                         // cm from the middle of the phone, + towards the bottom speaker
    static final int FULL = DISTANCES.length * SIDEWAYS.length + 1;
    static final String BACKGROUND = "background", EMPTY = "empty";

    /** One set: where the object stands, and the recordings to make, in order. */
    static final class Plan {
        final int distance, sideways;
        final List<String> trials = new ArrayList<>();

        Plan(int distance, int sideways) {
            this.distance = distance;
            this.sideways = sideways;
        }
    }

    final List<Plan> sets = new ArrayList<>();

    /** @param length how many sets: FULL, or fewer for a practice run */
    RunSheet(long seed, int length) {
        Random random = new Random(seed);
        List<int[]> positions = new ArrayList<>();
        for (int d : DISTANCES) for (int s : SIDEWAYS) positions.add(new int[]{d, s});
        Collections.shuffle(positions, random);
        positions = positions.subList(0, length - 1);

        List<Integer> places = new ArrayList<>();
        for (int i = 0; i < positions.size(); i++) places.add(i);
        Collections.shuffle(places, random);
        List<Integer> empties = places.subList(0, Math.max(1, positions.size() / 5));

        for (int i = 0; i < positions.size(); i++) sets.add(plan(positions.get(i), empties.contains(i), random));
        sets.add(plan(positions.get(0), false, random));
    }

    private static Plan plan(int[] position, boolean empty, Random random) {
        Plan p = new Plan(position[0], position[1]);
        List<String> objects = new ArrayList<>(Arrays.asList(Classifier.CLASSES));
        Collections.shuffle(objects, random);
        p.trials.add(BACKGROUND);
        p.trials.addAll(objects);
        if (empty) p.trials.add(1 + random.nextInt(objects.size() + 1), EMPTY);
        return p;
    }
}
