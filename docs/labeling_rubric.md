# Counting rules

For each image, count every object of each requested class. Count what you **see**, not what the prompt asked for.

## What counts

1. **Pictures of objects count.** Dogs printed on a cup, a painting of an elephant, a photo on a wall, a toy car: count them as the class they show.
   - Exception: if the picture or toy is itself one of the requested classes, count it only as that class. A teddy bear is a teddy bear, not a bear.
2. **Partly hidden, background, or small objects count** when **at least about 30% of the object is visible**, or when the **head of a living animal** (people included) is visible. A half-visible car at the edge counts; a sliver of bumper does not. An elephant head behind a wall counts.
3. **Malformed or fused objects count.**
   - A person with three arms is one person. A bicycle with a wrong number of wheels is one bicycle.
   - A hybrid counts as the class of its **head**: a dog's head on a human body is one dog and zero people.
   - Fused animals or people: count the heads.
   - Fused objects without heads (two cars melted together): count how many distinct objects a viewer would see.
4. **Reflections do not count**: the reflected object is already counted once.

## Class boundaries

Use what most people would call the object. Common cases:

| Class | Counts | Does not count |
|---|---|---|
| cup | mugs, teacups, coffee and paper cups | wine glasses, plain drinking glasses, bowls, jars, vases, plant pots |
| wine glass | stemmed glasses | tumblers, beer glasses, mugs |
| bowl | cereal or soup bowls, mixing bowls | plates, cups, pots |
| car | cars, SUVs, vans, taxis | pickup trucks and lorries (truck), buses |
| truck | pickups, lorries, delivery trucks | vans, buses |
| bicycle | bicycles | motorcycles, scooters |
| chair | chairs, stools, armchairs | couches, benches |
| couch | sofas, loveseats | armchairs, benches |
| dining table | any table set or used for a meal | desks, side tables with no food |
| potted plant | plant with its pot, one object | a pot with no plant, cut flowers in a vase |
| person | anyone, including babies and statues or pictures of people (rule 1) | mannequins' bare forms, shadows |
| teddy bear | teddy bears and similar plush toys | real bears |

If a case is not covered, decide what most people would say, apply it consistently, and add a note.

## When to tick "ambiguous"

Only when these rules do not settle the count:

- you cannot tell whether about 30% of an object is visible,
- you cannot tell which class something is,
- the scene is too crowded or blurry to count reliably.

Always enter your best count anyway, and **write a short note** saying what is unclear (for example "1 car might be 2 fused"). Notes are useful even when you do not tick ambiguous.
