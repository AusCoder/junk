#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOG_MALLOC_FAIL printf("malloc/realloc failed\n");

#define CHECK_MALLOC(ptr)                                                      \
  do {                                                                         \
    if (ptr == NULL) {                                                         \
      LOG_MALLOC_FAIL;                                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

typedef enum { TermTypeValue, TermTypeBinaryOp } TermType;

typedef struct {
  int value;
} TermValue;

typedef struct {
  char op;
} TermBinaryOp;

typedef struct {
  TermType type;
  union {
    TermValue value;
    TermBinaryOp op;
  };
} Term;

typedef struct {
  Term *terms;
  int size;
} ParsedTerms;

int parse_num(const char *digits, int *out) {
  int value = 0;
  for (int i = 0;; i++) {
    if (digits[i] == '\0') {
      break;
    }
    int dval = digits[i] - '0';
    if ((dval >= 0) && (dval <= 9)) {
      value = value * 10 + dval;
    } else {
      return -1;
    }
  }
  *out = value;
  return 0;
}

int parse_terms(char *input, Term **parsed_terms, int *size) {
  const char *delim = " ";
  int value, capacity;
  // Alloc terms
  capacity = 2;
  Term *terms = (Term *)malloc(capacity * sizeof(Term));
  CHECK_MALLOC(terms);

  // Parse terms
  int idx;
  for (idx = 0;; idx++) {
    char *tok = strtok(input, delim);
    input = NULL;
    if (tok == NULL) {
      break;
    }
    // Resize memory if needed
    if (idx >= capacity) {
      capacity *= 2;
      terms = realloc(terms, capacity * sizeof(Term));
      CHECK_MALLOC(terms);
    }
    // Parse token
    if (strcmp(tok, "+") == 0) {
      terms[idx].type = TermTypeBinaryOp;
      terms[idx].op.op = '+';
    } else if (strcmp(tok, "-") == 0) {
      terms[idx].type = TermTypeBinaryOp;
      terms[idx].op.op = '-';
    } else if (strcmp(tok, "*") == 0) {
      terms[idx].type = TermTypeBinaryOp;
      terms[idx].op.op = '*';
    } else if (strcmp(tok, "/") == 0) {
      terms[idx].type = TermTypeBinaryOp;
      terms[idx].op.op = '/';
    } else if (parse_num(tok, &value) >= 0) {
      terms[idx].type = TermTypeValue;
      terms[idx].value.value = value;
    } else {
      printf("unknown tok \"%s\"\n", tok);
      free(terms);
      return -1;
    }
  }
  *parsed_terms = terms;
  *size = idx;
  return 0;
}

int evaluate(const Term *terms, int size, int *outvalue) {
  int retval = 0;
  int stacksize = 1;
  int stackptr = 0;
  int *stack = (int *)malloc(stacksize * sizeof(int));
  CHECK_MALLOC(stack);

  for (int idx = 0; idx < size; idx++) {
    // Resize stack if needed
    if (stackptr >= stacksize) {
      stacksize *= 2;
      stack = realloc(stack, stacksize * sizeof(int));
      CHECK_MALLOC(stack);
    }
    // Evaluate using the stack
    switch (terms[idx].type) {
    case TermTypeValue:
      stack[stackptr] = terms[idx].value.value;
      stackptr++;
      break;
    case TermTypeBinaryOp:
      if (stackptr <= 1) {
        printf("Not enough terms to evaluate binary op\n");
        retval = -1;
        goto done;
      }
      stackptr--;
      int arg2 = stack[stackptr];
      stackptr--;
      int arg1 = stack[stackptr];
      // Evaluate binary terms
      char binop = terms[idx].op.op;
      switch (binop) {
      case '+':
        stack[stackptr] = arg1 + arg2;
        stackptr++;
        break;
      case '-':
        stack[stackptr] = arg1 - arg2;
        stackptr++;
        break;
      case '*':
        stack[stackptr] = arg1 * arg2;
        stackptr++;
        break;
      case '/':
        stack[stackptr] = arg1 / arg2;
        stackptr++;
        break;
      default:
        printf("unknown binary op: %c\n", binop);
        retval = -1;
        goto done;
      }
      break;
    default:
      printf("unknown term type");
      retval = -1;
      goto done;
    }
  }
  if (stackptr == 1) {
    *outvalue = stack[0];
  } else {
    printf("invalid reverse polish expression passed, stackptr not at 1\n");
    retval = -1;
  }
done:
  free(stack);
  return retval;
}

int main(int argc, char **argv) {
  if (argc <= 1) {
    printf("usage: calc <reverse-polish-expr>\n");
    return 1;
  }

  Term *terms = NULL;
  int size, value;
  if (parse_terms(argv[1], &terms, &size) < 0) {
    return 1;
  }
  if (evaluate(terms, size, &value) < 0) {
    free(terms);
    return 1;
  }
  printf("calculated value: %d\n", value);
  free(terms);
  return 0;
}
