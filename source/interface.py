def leiaInt(msg):
	while True:
		try:
			n = int(input(msg))
		except (ValueError, TypeError):
			print('ERRO: por favor, digite um número inteiro válido.')
		except (KeyboardInterrupt):
			print('Usuário preferiu não digitar esse número.')
			return 0
		else:
			return n

def linha(tam=42):
	return '-' * tam

def cabecalho(txt):
	print('\n')
	print(linha())
	print(txt.center(42))
	print(linha())


def menu(lista):
	cabecalho('DIGITE A OPÇÃO DESEJADA')
	c = 1
	for item in lista:
		print(f'{c} - {item}')
		c += 1
	print(linha())
	opc = leiaInt('Sua Opção:')
	return opc


