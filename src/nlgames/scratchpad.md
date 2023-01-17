## Representing repetition: classical

strategy vectors currently: [a,b,c,d] - index representing the question, value representing the answer.
strategy vectors with 2-fold parallel repetition:
[[a1,b1,c1,d1],
 [a2,b2,c2,d2]]
with each row representing a new answer-question pair.

before: q1 values, now: q1*reps

Generate long vector, reshape??

:ij
logical_and


## The above doesn't seem to work very well. Instead:

One long strategy vector.
Basically _n_ strategy vectors attached to one another.


## TODO
Predicate matrix needs more dimensions??