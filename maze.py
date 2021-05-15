import sys
from enum import Enum
from typing import Dict, Type, TypeVar, Tuple, List, Optional

import numpy as np
from numpy import ndarray

M = TypeVar('M', bound='Maze')


class Movement(Enum):
    """
    This enum contains the four possible directions (north, east, south and west).
    Additionally the opposite direction (north => south, east => west, south => north and west => east)
    is possible to get.
    """
    NORTH: str = 'N'
    EAST: str = 'E'
    SOUTH: str = 'S'
    WEST: str = 'W'

    @staticmethod
    def get_all() -> List['Movement']:
        # Return all possible movements
        return [Movement.NORTH, Movement.EAST, Movement.SOUTH, Movement.WEST]

    def get_opposite(self) -> 'Movement':
        # Get opposite of the direction (e.g. south => north)
        if self == Movement.NORTH:
            return Movement.SOUTH
        if self == Movement.EAST:
            return Movement.WEST
        if self == Movement.SOUTH:
            return Movement.NORTH
        return Movement.EAST


class Maze(object):
    # List of all possible movements (North, East, South, West)
    MOVEMENTS: List[Movement] = Movement.get_all()

    # Some constants for the replacement with the chars in the maze file
    START_POINT: int = 2
    GOAL_POINT: int = 3
    EMPTY: int = 1
    WALL: int = 0
    NORTH: int = 4
    EAST: int = 5
    SOUTH: int = 6
    WEST: int = 7

    # Maps char to integer for doing the work in the algorithm
    CHAR_MAP: Dict[str, int] = {
        "*": WALL,
        " ": EMPTY,
        "A": START_POINT,
        "B": GOAL_POINT,
        "N": NORTH,
        "E": EAST,
        "S": SOUTH,
        "W": WEST
    }

    # Container for the two-dimensional list
    maze: ndarray

    # Points in the maze representing current position and position of
    # goal
    current_pos: Tuple[int, int] = (0, 0)
    goal_pos: Tuple[int, int] = (0, 0)

    # Store of the path which has been followed
    path: List[Tuple[int, int]] = []
    # Store of all the movements which have been made
    # Those movements are in relation to the points in the path List
    movements: List[Movement] = []

    def __init__(self, maze: ndarray) -> None:
        self.maze = maze

    def get_start_point(self) -> Tuple[int, int]:
        return self.__get_point(self.START_POINT)

    def get_goal_point(self) -> Tuple[int, int]:
        return self.__get_point(self.GOAL_POINT)

    def solve(self) -> bool:
        """
        The algorithm searches depth first for a possible path.
        1. Get the start point.
        2. Search for possible movement in the order of North, East, South, West.
           Exceptional case: no possible movement found: Go back one point and go another way.
        3. Follow possible movement and repeat steps 2 - 3.

        Not possible movements (in point 2) are:
        - Cycles (if the point is already within the path)
        - Walls

        Stop of the algorithm in one of those cases:
        1) Path reaches the goal point.
        2) Had as many exceptional cases in step 2 that it got back to the start point.

        :return: bool TRUE if there is a result, FALSE if there is no path found from A to B
        """
        self.current_pos = self.get_start_point()
        self.goal_pos = self.get_goal_point()

        # loop as long as we don't reach goal position
        while True:
            # Step 1: get the next possible movement
            next_movement = self.__get_next_possible_movement()
            while next_movement is None:
                # Step 2: Exceptional case: no possible movement anymore
                # Step back and repeat Step 2
                self.path.pop()
                if len(self.path) == 0:
                    # Stop case 2: no solution found as we are at the beginning again
                    return False

                # Step 1: get next possible movement after step back in exceptional case.
                self.current_pos = self.path[-1]
                last_movement = self.movements.pop()
                next_movement = self.__get_next_possible_movement(
                    last_movement)

            # Step 3: Follow possible movement
            self.current_pos = \
                self.__get_point_by_movement(
                    self.current_pos, next_movement
                )

            # Stop case 1: Goal reached
            if self.current_pos == self.goal_pos:
                return True

            self.movements.append(next_movement)
            self.path.append(self.current_pos)

    @classmethod
    def from_txt(cls: Type[M], file: str) -> M:
        # As per the suggestion on
        # https://stackoverflow.com/questions/19792112/static-method-returning-an-instance-of-its-class
        # I needed to put Maze into quotes as it is not recognized here
        try:
            f = open(file, "r")
        except FileNotFoundError:
            print("Could not open file {}".format(file),
                  file=sys.stderr)
            exit(1)

        lines = f.read().splitlines()
        f.close()

        matrix = []
        for line in lines:
            for i, j in Maze.CHAR_MAP.items():
                line = line.replace(i, str(j))
            elements = list(map(int, list(line)))
            matrix.append(np.array(elements))
        maze = np.array(matrix)
        return cls(maze)

    @staticmethod
    def get_maze(maze: ndarray) -> str:
        output = ''
        for line in maze:
            line = list(map(str, list(line)))
            str_line = ''.join(line)
            for i, j in Maze.CHAR_MAP.items():
                str_line = str_line.replace(str(j), i)
            output += str_line + "\n"
        return output

    def get_maze_with_movements(self) -> str:
        """
        Add the path to the maze by adding the integer of the movement to
        the maze at the correct place.
        :return: string
        """
        maze = self.maze.copy()
        movements = self.movements.copy()
        for point in self.path:
            x, y = point
            maze[y][x] = self.CHAR_MAP[movements.pop(0).value]
        return self.get_maze(maze)

    def get_path(self) -> str:
        return ''.join([movement.value for movement in self.movements])

    def __str__(self) -> str:
        return self.get_maze(self.maze)

    @staticmethod
    def __get_point_by_movement(point: Tuple[int, int],
                                movement: Movement) -> Tuple[int, int]:
        # calculate the next point based on the direction to go
        x, y = point
        if movement == Movement.NORTH:
            return x, y - 1
        if movement == Movement.EAST:
            return x + 1, y
        if movement == Movement.SOUTH:
            return x, y + 1
        if movement == Movement.WEST:
            return x - 1, y
        raise Exception("Unknown movement {}".format(movement))

    def __get_point(self, point: int) \
            -> Tuple[int, int]:
        # Search for start or goal point
        result = np.where(self.maze == point)
        y, x = int(result[0]), int(result[1])
        return x, y

    def __is_valid_point(self, point: Tuple[int, int]) -> bool:
        x, y = point
        if point in self.path:
            # Cycle detected: no possible movement
            return False
        # Only EMPTY and GOAL_POINT is a possible movement
        return self.maze[y, x] and \
               self.maze[y, x] in [self.EMPTY, self.GOAL_POINT]

    def __get_next_possible_movement(
            self, already_tried_movement: Optional[Movement] = None
    ) -> Optional[Movement]:
        # find possible movement starting from the already_tried_movement (this is the case when stepping back)
        movements = self.MOVEMENTS
        if already_tried_movement:
            # remove the already tried movements
            index_last_movement = self.MOVEMENTS.index(
                already_tried_movement)
            movements = self.MOVEMENTS[index_last_movement + 1:]

        for movement in movements:
            # loop through all movements and check whether there is a valid path to follow
            if len(self.movements) > 0:
                # the path back is not allowed
                if movement == self.movements[-1].get_opposite:
                    continue

            next_point = self.__get_point_by_movement(
                self.current_pos, movement
            )
            if self.__is_valid_point(next_point):
                return movement
        return None


def main(file: str) -> None:
    # initialize Maze object by text file
    maze = Maze.from_txt(file)

    # start to find a path through the maze
    if maze.solve():
        # maze has a solution
        # print solved maze
        print('Solve maze:')
        print(maze.get_maze_with_movements())

        # print path as string
        print('Path: {}'.format(maze.get_path()))
    else:
        # maze has no solution
        print("No solution could be found")


if __name__ == "__main__":
    # check whether argument is set
    if sys.argv[1] is None:
        print("Missing file parameter", file=sys.stderr)
        exit(1)

    main(sys.argv[1])
