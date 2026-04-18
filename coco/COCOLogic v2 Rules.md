Names:
1. Signal and Ride: A traffic light and one of: bicycle, bus or train 
2. Double Serving: Exactly two of the following categories are present (bottle, cup, pizza)
3. Herd Alone: The image includes two or more animals (of the same type) (cow, elephant, sheep) and no people
4. Dog or Car: Either a dog or a car
5. Three of a kind: exactly three bowls or exactly three cups
6. Car Majority: more cars than trucks (each appear at least once)
7. Empty Seat: furniture (couch or chair) and no person
8. Single Mode Traffic: exactly one of the following categories: bicycle, motorcycle, car, bus
9. Personal Transport: a person and either a bicycle or a car (but not both)
10. Surf Trip: the same number of persons and surfboards (at least one each)

Components and Hard Negatives:
**Rule 1**
P1: traffic light + bicycle
P2: traffic light + bus
P3: traffic light + train
N1: no traffic light, bicycle exists
N2: no traffic light, bus exists
N3: no traffic light, train exists
N4: traffic light exists, none of bicycle/bus/train

**Rule 2**
P1: bottle + cup
P2: bottle + pizza
P3: cup + pizza
N1: bottle only
N2: cup only
N3: pizza only
N4: bottle + cup + pizza


**Rule 3**
P1: two cows and no person
P2: two elephants and no person
P3: two sheep and no person
N1: exactly 1 elephant and no persons
N2: exactly 1 cow and no persons
N3: exactly 1 sheep and no persons
N4: at least 2 elephants and at least one person
N5: at least 2 cows and at least one person
N6: at least 2 sheep and at least one person

**Rule 4**
P1: dog present and no car
P2: car present and no dog
N1: dog exists and car exists
N2: neither a dog nor a car exists

**Rule 5**
P1: exactly 3 bowls
P2: exactly 3 cups
N1: 1 - 2 cups
N2: 4 or more cups
N3: 1 - 2 bowls
N4: 4 or more bowls

**Rule 6**
P1:  more cars than trucks, and both appear at least once
N1: at least 1 car and no trucks
N2: equal number of cars and trucks; both appear
N3: more trucks than cars

**Rule 7**
P1: couch and no person
P2: chair and no person
N1: couch + person
N2: chair + person

**Rule 8**
P1: bicycle only
P2: motorcycle only
P3: car only
P4: bus only
N1: bicycle + motorcycle only
N2: bicycle + car only
N3: bicycle + bus only
N4: motorcycle + car only
N5: motorcycle + bus only
N6: car + bus only

**Rule 9**
P1:  person and bicycle, but not car
P2: person and car, but not bicycle
N1: person + bicycle + car
N2: person + no bicycle + no car
N3: bicycle + no car + no person
N4: car + no bicycle + no person

**Rule 10**
P1:  Same number of persons and surfboards, both appear at least once
N1: at least 1 surfboard and no persons
N2: at least 1 person and no surfboards
N3: more surfboards than persons, both appear 
N4: more persons than surfboards, both appear




Signal and Ride & \texttt{traffic light} AND one of: \{\texttt{bicycle}, \texttt{bus}, \texttt{train}\} \\
\midrule
Double Serving & At least one object of exactly two categories of \{\texttt{bottle, cup, pizza}\}\\
\midrule
Herd Alone & Two or more animals of the same type (\{\texttt{cow, elephant, sheep}\} AND no \texttt{person} \\
\midrule
Dog or Car & \texttt{dog} XOR \texttt{car} \\
\midrule
Three of a kind & Exactly three \{\texttt{bowls}\} OR exactly three \{\texttt{cups}\} \\
\midrule
Car Majority & More \texttt{cars} than \texttt{trucks} AND at least one of each \\
\midrule
Empty Seat & (\texttt{couch} OR \texttt{chair}) AND no \texttt{person} \\
\midrule
Single Mode Traffic & Exactly one of the following categories \{\texttt{bicycle, motorcycle, car, bus}\} \\
\midrule
Personal Transport & \texttt{person} AND (\texttt{bicycle} XOR \texttt{car}) \\
\midrule
Surf Trip & Exactly as many \texttt{person} as \{\texttt{surf board}\} \\


Regeln mit Counts:
10, 6, 5