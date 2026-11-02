package dev.phonesonar;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Checks what RunSheet promises, over many seeds. Stops with an error at the first thing that does not hold. */
public class RunSheetCheck {
    static void check(boolean ok, String what) {
        if (!ok) throw new AssertionError(what);
    }

    static void plan(RunSheet.Plan p, long seed) {
        check(p.trials.get(0).equals(RunSheet.BACKGROUND), "the empty scene comes first, seed " + seed);
        check(p.trials.containsAll(Arrays.asList(Classifier.CLASSES)), "both objects in every set, seed " + seed);
        check(p.trials.size() == (p.trials.contains(RunSheet.EMPTY) ? 4 : 3), "nothing extra in a set, seed " + seed);
    }

    public static void main(String[] args) {
        int firstFish = 0, sets = 0;
        int[] emptyAt = new int[4];
        int positions = RunSheet.DISTANCES.length * RunSheet.SIDEWAYS.length;
        for (long seed = 1; seed <= 2000; seed++) {
            List<RunSheet.Plan> full = new RunSheet(seed, RunSheet.FULL).sets;
            check(full.size() == positions + 1, "every position and a repeat");
            Set<String> seen = new HashSet<>();
            int empties = 0;
            for (int i = 0; i < positions; i++) {
                RunSheet.Plan p = full.get(i);
                plan(p, seed);
                seen.add(p.distance + "/" + p.sideways);
                int at = p.trials.indexOf(RunSheet.EMPTY);
                if (at > 0) {
                    empties++;
                    emptyAt[at]++;
                }
                sets++;
                if (p.trials.indexOf("fish") < p.trials.indexOf("bobblehead")) firstFish++;
            }
            check(seen.size() == positions, "every position once, seed " + seed);
            check(empties == Math.max(1, positions / 5), "about one set in five has an empty trial, seed " + seed);
            RunSheet.Plan last = full.get(positions), first = full.get(0);
            plan(last, seed);
            check(last.distance == first.distance && last.sideways == first.sideways, "the last set repeats the first, seed " + seed);
            check(!last.trials.contains(RunSheet.EMPTY), "no empty trial in the repeat, seed " + seed);

            List<RunSheet.Plan> again = new RunSheet(seed, RunSheet.FULL).sets;
            for (int i = 0; i < full.size(); i++) check(again.get(i).trials.equals(full.get(i).trials), "the same seed gives the same sheet");

            List<RunSheet.Plan> practice = new RunSheet(seed, 3).sets;
            check(practice.size() == 3, "a practice run is as long as asked");
            check(practice.get(0).trials.contains(RunSheet.EMPTY) || practice.get(1).trials.contains(RunSheet.EMPTY), "a practice run has an empty trial");
        }
        System.out.printf("run sheet: 2000 seeds hold; fish first in %.0f%% of sets; empty trial at places 2, 3, 4: %d, %d, %d%n",
                100.0 * firstFish / sets, emptyAt[1], emptyAt[2], emptyAt[3]);
    }
}
